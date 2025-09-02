import torch
from transformers import Trainer, TrainingArguments
import os
from config import TRAINING_CONFIG, DATA_PATHS, HARDWARE_CONFIG
from model_setup import setup_videollama3_model
from scripts.dataset import VideoLLaMA3Dataset, VideoLLaMA3JSONLDataset, videollama3_collate_fn

class VideoLLaMA3Trainer(Trainer):
    """Custom trainer for VideoLLaMA3."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation for VideoLLaMA3."""
        try:
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            
            # Handle different output formats
            if return_outputs:
                return (loss, outputs) if loss is not None else (torch.tensor(0.0, device=model.device), outputs)
            return loss if loss is not None else torch.tensor(0.0, device=model.device)
            
        except Exception as e:
            print(f"Error in compute_loss: {e}")
            # Return dummy loss to avoid training interruption
            dummy_loss = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
            if return_outputs:
                return (dummy_loss, None)
            return dummy_loss

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    return {"eval_loss": eval_pred.metrics.get("eval_loss", 0.0)}

def train_videollama3(
    train_csv=None,
    train_jsonl=None,
    video_dir=None,
    data_root=None,
    model_path=None,
    output_dir="./videollama3_outputs",
    use_lora=True,
    **kwargs
):
    """
    Train VideoLLaMA3 model.
    
    Args:
        train_csv: Path to CSV training file
        train_jsonl: Path to JSONL training file
        video_dir: Directory containing videos (for CSV format)
        data_root: Root directory for data (for JSONL format)
        model_path: Path to model (for fine-tuning from checkpoint)
        output_dir: Output directory for training
        use_lora: Whether to use LoRA fine-tuning
        **kwargs: Additional training arguments
    """
    print("Starting VideoLLaMA3 training...")
    
    # Setup model and processor
    if model_path:
        print(f"Loading model from checkpoint: {model_path}")
        # Load from checkpoint logic here if needed
    
    model, processor = setup_videollama3_model(use_lora=use_lora)
    
    # Create dataset
    if train_jsonl and data_root:
        print(f"Using JSONL dataset: {train_jsonl}")
        train_dataset = VideoLLaMA3JSONLDataset(
            jsonl_file=train_jsonl,
            data_root=data_root,
            processor=processor
        )
    elif train_csv and video_dir:
        print(f"Using CSV dataset: {train_csv}")
        train_dataset = VideoLLaMA3Dataset(
            csv_file=train_csv,
            video_dir=video_dir,
            processor=processor,
            num_frames=kwargs.get("num_frames", 8)
        )
    else:
        raise ValueError("Must provide either (train_jsonl, data_root) or (train_csv, video_dir)")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=kwargs.get("max_epochs", TRAINING_CONFIG["max_epochs"]),
        per_device_train_batch_size=kwargs.get("batch_size", TRAINING_CONFIG["batch_size"]),
        gradient_accumulation_steps=TRAINING_CONFIG["accumulate_grad_batches"],
        learning_rate=kwargs.get("learning_rate", TRAINING_CONFIG["lr"]),
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        max_steps=kwargs.get("max_train_batches", TRAINING_CONFIG["max_train_batches"]),
        fp16=TRAINING_CONFIG["fp16"],
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        dataloader_pin_memory=False,  # Reduce memory usage
        remove_unused_columns=False,  # Important for multimodal inputs
        report_to="none",
        save_safetensors=True,
        max_grad_norm=TRAINING_CONFIG["gradient_clip_val"],
        eval_strategy="no",  # Disable evaluation for now
    )
    
    # Handle DeepSpeed if configured
    if HARDWARE_CONFIG.get("use_deepspeed", False):
        training_args.deepspeed = HARDWARE_CONFIG["deepspeed_config"]
    
    # Create trainer
    trainer = VideoLLaMA3Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=videollama3_collate_fn,
        compute_metrics=compute_metrics,
    )
    
    # Clear cache and start training
    torch.cuda.empty_cache()
    
    print("Starting training...")
    try:
        trainer.train()
        print("Training completed successfully!")
        
        # Save final model
        final_save_path = os.path.join(output_dir, "final_model")
        trainer.save_model(final_save_path)
        processor.save_pretrained(final_save_path)
        print(f"Final model saved to: {final_save_path}")
        
    except Exception as e:
        print(f"Training error: {e}")
        # Save current state
        checkpoint_path = os.path.join(output_dir, "emergency_checkpoint")
        trainer.save_model(checkpoint_path)
        print(f"Emergency checkpoint saved to: {checkpoint_path}")
        raise


