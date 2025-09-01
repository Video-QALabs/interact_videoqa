# scripts/train.py
# Clean training loop for Qwen2.5-VL following the HF cookbook pipeline
# messages -> apply_chat_template -> process_vision_info -> processor(...)
# and with strict guards to avoid passing [[]] into processor(images=...)

import os
import gc
from typing import Dict, Any, List, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import amp
from tqdm import tqdm

from scripts.config import (
    TRAINING_CONFIG,
    MEMORY_CONFIG,
    setup_memory_optimized_environment,
    MemoryMonitor,
)
from qwen_vl_utils import process_vision_info as _pvi


# --------------------------- utilities ---------------------------

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def check_gpu_memory(stage: str = "") -> Tuple[float, float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    max_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
    if stage:
        print(f"\n[GPU Memory {stage}]")
        print(f"Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Max: {max_alloc:.2f}GB")
    return allocated, reserved, max_alloc


def print_batch_info(batch: Dict[str, Any]):
    print("\n=== Batch Debug Info ===")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            mb = v.numel() * v.element_size() / (1024 * 1024)
            print(f"{k}: {tuple(v.shape)} | {v.dtype} | {mb:.1f}MB")
        elif isinstance(v, list):
            print(f"{k}: List with {len(v)} items")
        else:
            print(f"{k}: {type(v)}")


def extract_loss_from_outputs(outputs: Any, tag: str) -> Optional[torch.Tensor]:
    if outputs is None:
        print(f"{tag}: model returned None")
        return None

    if hasattr(outputs, "loss") and outputs.loss is not None:
        loss = outputs.loss
    elif isinstance(outputs, dict) and "loss" in outputs:
        loss = outputs["loss"]
    elif isinstance(outputs, tuple) and len(outputs) > 0:
        loss = outputs[0]
    else:
        print(f"{tag}: no loss in outputs")
        return None

    if isinstance(loss, torch.Tensor) and loss.ndim > 0:
        loss = loss.mean()
    if not isinstance(loss, torch.Tensor):
        loss = torch.tensor(loss, dtype=torch.float32)

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"{tag}: invalid loss value {loss}")
        return None
    return loss


# ----------------- messages -> processor inputs ------------------

def _normalize_media_arg(x):
    """
    Convert empty lists ([], [[]]) to None.
    Leave non-empty lists as-is.
    """
    if x is None:
        return None
    if isinstance(x, list):
        if len(x) == 0:
            return None
        if len(x) == 1 and isinstance(x[0], list) and len(x[0]) == 0:
            return None
    return x


def _safe_process_vision_info(messages: List[Dict[str, Any]]) -> Tuple[Optional[list], Optional[list], dict]:
    """
    Wrap qwen_vl_utils.process_vision_info and guarantee that we never return
    empty image/video lists (we return None instead).
    """
    # Newer utils may return (images, videos, kwargs); older often (images, videos)
    try:
        out = _pvi(messages, return_video_kwargs=True)  # type: ignore
        if isinstance(out, tuple) and len(out) == 3:
            img, vid, vkw = out
        elif isinstance(out, tuple) and len(out) == 2:
            img, vid = out
            vkw = {}
        else:
            # Unexpected; force text-only
            return None, None, {}
    except TypeError:
        # Older signature without return_video_kwargs
        img, vid = _pvi(messages)
        vkw = {}

    img = _normalize_media_arg(img)
    vid = _normalize_media_arg(vid)
    if not isinstance(vkw, dict):
        vkw = {}

    return img, vid, vkw


def build_inputs_from_messages(
    batch: Dict[str, Any],
    processor,
    device: torch.device,
    for_labels: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    HF cookbook-compliant path:
      - batch["messages"] is a list of per-sample message lists
      - For training, we create text with add_generation_prompt=False (so labels align to the whole sequence)
      - We only pass images/videos to processor if they are non-empty; otherwise pass None.
    """
    msg_list = batch.get("messages", None)
    if not isinstance(msg_list, (list, tuple)) or len(msg_list) == 0:
        return None

    # Validate/collect samples
    valid_msgs: List[List[Dict[str, Any]]] = []
    for msgs in msg_list:
        if isinstance(msgs, list) and len(msgs) > 0:
            valid_msgs.append(msgs)
    if not valid_msgs:
        return None

    texts: List[str] = []
    per_sample_images: List = []
    per_sample_videos: List = []
    # We’ll merge video kwargs across samples only when identical; otherwise use defaults
    merged_video_kwargs: dict = {}

    # Build prompt & media per sample
    for i, messages in enumerate(valid_msgs):
        # NO generation prompt during training
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

        img_inp, vid_inp, vkw = _safe_process_vision_info(messages)
        per_sample_images.append(img_inp)   # may be None
        per_sample_videos.append(vid_inp)   # may be None

        if i == 0 and isinstance(vkw, dict):
            merged_video_kwargs = vkw

    # Prepare kwargs for processor: only pass media keys if there’s at least one non-None
    kwargs: Dict[str, Any] = {"text": texts, "padding": True, "return_tensors": "pt"}
    if any(item is not None for item in per_sample_images):
        kwargs["images"] = per_sample_images
    if any(item is not None for item in per_sample_videos):
        kwargs["videos"] = per_sample_videos
        kwargs.update(merged_video_kwargs or {})

    # Build tensors
    inputs = processor(**kwargs)

    # Labels from input_ids (mask pads to -100)
    if for_labels and "input_ids" in inputs:
        labels = inputs["input_ids"].clone().to(torch.long)
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels

    # Move to device
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device, non_blocking=True)

    return inputs


def prepare_memory_efficient_inputs(
    batch: Dict[str, Any],
    device: torch.device,
    processor=None,
) -> Optional[Dict[str, Any]]:
    """
    Preferred path: rebuild from messages (chat + vision).
    If that fails, skip the batch instead of trying half-baked tensor paths.
    """
    if processor is not None and "messages" in batch:
        try:
            return build_inputs_from_messages(batch, processor, device, for_labels=True)
        except Exception as e:
            print(f"messages-based build failed, skipping batch: {e}")
            return None

    # If there’s no messages, we don’t try to shoehorn tensors; skip.
    req = [k for k in ("input_ids", "attention_mask") if k not in batch]
    if req:
        print(f"Missing required keys: {req}")
        return None

    # Move minimal text-only tensors (rare path)
    inputs = {}
    for k in ("input_ids", "attention_mask"):
        v = batch[k]
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device, non_blocking=True)
    if inputs:
        labels = inputs["input_ids"].clone().to(torch.long)
        labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        return inputs
    return None


# ------------------------- training/eval -------------------------

def handle_runtime_error(error: Exception, step_num: int, optimizer: torch.optim.Optimizer, monitor: MemoryMonitor):
    msg = str(error).lower()
    if "out of memory" in msg:
        print(f"OOM at step {step_num} — clearing cache and continuing")
        monitor.log_oom()
        optimizer.zero_grad(set_to_none=True)
        clear_gpu_memory()
    else:
        print(f"Runtime error at step {step_num}: {error}")
        optimizer.zero_grad(set_to_none=True)
    monitor.log_failure()


def generate_memory_efficient_sample(model: nn.Module, processor, batch: Dict[str, Any]):
    """
    Quick sample generation using the first item’s messages with add_generation_prompt=True.
    """
    model.eval()
    try:
        if "messages" not in batch or not batch["messages"]:
            print("\nSample Generation skipped: no 'messages' in batch.")
            return

        device = next(model.parameters()).device
        messages = batch["messages"][0]

        # Build generation text
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Media
        image_inputs, video_inputs, video_kwargs = _safe_process_vision_info(messages)
        gen_kwargs = {"text": [text_prompt], "padding": True, "return_tensors": "pt"}
        if image_inputs is not None:
            gen_kwargs["images"] = image_inputs
        if video_inputs is not None:
            gen_kwargs["videos"] = video_inputs
            gen_kwargs.update(video_kwargs or {})

        infer_inputs = processor(**gen_kwargs)
        for k, v in list(infer_inputs.items()):
            if isinstance(v, torch.Tensor):
                infer_inputs[k] = v.to(device)

        with torch.no_grad():
            # NOTE: newer transformers warn about 'temperature' on some configs; omit if noisy
            generated_ids = model.generate(
                **infer_inputs,
                max_new_tokens=64,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                use_cache=False,
                pad_token_id=getattr(processor.tokenizer, "pad_token_id", None)
                or getattr(processor.tokenizer, "eos_token_id", None),
                eos_token_id=getattr(processor.tokenizer, "eos_token_id", None),
            )

        in_len = infer_inputs["input_ids"].shape[1]
        new_tokens = generated_ids[:, in_len:]
        response = processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        gt = batch.get("ground_truth", ["N/A"])[0]
        print("\nSample Generation:")
        print(f"Generated: {response[:120]}{'...' if len(response) > 120 else ''}")
        print(f"Expected:  {gt[:120]}{'...' if len(gt) > 120 else ''}")

    except Exception as e:
        print(f"Generation error: {e}")
    finally:
        model.train()
        clear_gpu_memory()


def memory_efficient_evaluate(model: nn.Module, processor, eval_loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    max_eval_batches = min(10, len(eval_loader))

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= max_eval_batches:
                break
            try:
                if i % 3 == 0:
                    clear_gpu_memory()

                inputs = prepare_memory_efficient_inputs(batch, device, processor=processor)
                if not inputs:
                    continue

                if device.type == "cuda":
                    with amp.autocast("cuda"):
                        outputs = model(**inputs)
                        loss = extract_loss_from_outputs(outputs, f"eval_{i}")
                else:
                    outputs = model(**inputs)
                    loss = extract_loss_from_outputs(outputs, f"eval_{i}")

                if loss is not None:
                    total_loss += float(loss.detach())
                    total_batches += 1

            except Exception as e:
                print(f"Eval batch {i} failed: {e}")
                continue

    if total_batches > 0:
        print(f"\n>>> Evaluation: avg loss {total_loss / total_batches:.4f} over {total_batches} batches")
    else:
        print(">>> Evaluation: no successful batches")

    model.train()
    clear_gpu_memory()


def enhanced_memory_training_loop(
    model: nn.Module,
    processor,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader],
    device: torch.device,
):
    setup_memory_optimized_environment()

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["lr"],
        weight_decay=TRAINING_CONFIG.get("weight_decay", 0.01),
        eps=1e-8,
        betas=(0.9, 0.95),
    )

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    if scaler:
        print("✓ Mixed precision training enabled")

    num_epochs = TRAINING_CONFIG["max_epochs"]
    accumulation_steps = TRAINING_CONFIG["accumulate_grad_batches"]
    max_train_batches = TRAINING_CONFIG["max_train_batches"]
    eval_interval = TRAINING_CONFIG["eval_interval"]
    gradient_clip_val = TRAINING_CONFIG["gradient_clip_val"]
    empty_cache_freq = MEMORY_CONFIG.get("empty_cache_frequency", 5)
    logging_steps = TRAINING_CONFIG.get("logging_steps", 10)

    monitor = MemoryMonitor()

    print("\n>>> Starting Memory-Optimized Training")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Cache clearing frequency: {empty_cache_freq}")
    print(f"Max train batches: {max_train_batches}")

    model.train()
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        epoch_loss = 0.0
        success = 0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            if (i + 1) > max_train_batches:
                break

            try:
                if i == 0:
                    print_batch_info(batch)
                    check_gpu_memory("Before processing")

                if i % empty_cache_freq == 0:
                    clear_gpu_memory()

                model_inputs = prepare_memory_efficient_inputs(batch, device, processor=processor)
                if not model_inputs:
                    req = ["input_ids", "attention_mask"]
                    print(f"Missing required keys: {req}")
                    print(f"Skipping batch {i}: invalid inputs")
                    continue

                try:
                    if scaler and device.type == "cuda":
                        with amp.autocast("cuda"):
                            outputs = model(**model_inputs)
                            loss = extract_loss_from_outputs(outputs, f"step_{i}")
                            if loss is None:
                                raise RuntimeError("loss is None")
                            loss = loss / accumulation_steps
                        scaler.scale(loss).backward()
                    else:
                        outputs = model(**model_inputs)
                        loss = extract_loss_from_outputs(outputs, f"step_{i}")
                        if loss is None:
                            raise RuntimeError("loss is None")
                        loss = loss / accumulation_steps
                        loss.backward()

                    step_loss = float(loss.detach()) * accumulation_steps
                    epoch_loss += step_loss
                    success += 1
                    monitor.log_success()

                    if i % logging_steps == 0:
                        print(f"Step {i}: loss = {step_loss:.4f}")

                except RuntimeError as e:
                    handle_runtime_error(e, i, optimizer, monitor)
                    continue

                if (i + 1) % accumulation_steps == 0:
                    try:
                        if scaler and device.type == "cuda":
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    except Exception as opt_e:
                        print(f"Optimizer step failed at {i}: {opt_e}")
                        optimizer.zero_grad(set_to_none=True)

                if (i + 1) % eval_interval == 0 and success > 0:
                    avg = epoch_loss / success
                    print(f"\n[Step {i+1}] Running avg loss: {avg:.4f}")
                    check_gpu_memory("During training")
                    # quick sample
                    try:
                        generate_memory_efficient_sample(model, processor, batch)
                    except Exception as gen_e:
                        print(f"Sample generation failed: {gen_e}")

            except Exception as e:
                print(f"Batch {i} failed completely: {e}")
                monitor.log_failure()
                optimizer.zero_grad(set_to_none=True)
                clear_gpu_memory()
                continue

        if success > 0:
            print(f"\n=== Epoch {epoch+1} Summary ===")
            print(f"Average Loss: {epoch_loss / success:.4f} | Successful batches: {success}")
        else:
            print("\nNo successful batches this epoch.")

        clear_gpu_memory()
        gc.collect()

        if eval_loader is not None and success > 0:
            try:
                print("\n>>> Evaluation ...")
                memory_efficient_evaluate(model, processor, eval_loader, device)
            except Exception as eval_e:
                print(f"Evaluation failed: {eval_e}")

    print("\n>>> Training Complete!")
    stats = MemoryMonitor().get_stats()
    print(f"Peak Memory: {stats['peak_memory_gb']:.2f}GB")
    print(f"OOM Events: {stats['oom_events']}")

    return model


# ------------------- public entrypoint -------------------

def train_model(model, processor, train_loader, eval_loader, device):
    """
    Backward-compatible wrapper.
    """
    return enhanced_memory_training_loop(model, processor, train_loader, eval_loader, device)
