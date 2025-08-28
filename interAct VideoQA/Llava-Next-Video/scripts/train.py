import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from config import TRAINING_CONFIG, GENERATION_CONFIG

def train_model(model, processor, train_loader, eval_loader, device):
    """
    Train the LLaVA-Next-Video model.
    
    Args:
        model: The LLaVA model
        processor: The processor
        train_loader: Training data loader
        eval_loader: Evaluation data loader  
        device: Device to train on
    
    Returns:
        model: Trained model
    """
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=TRAINING_CONFIG["lr"],
        weight_decay=TRAINING_CONFIG.get("weight_decay", 0.01)
    )
    
    # Training parameters
    num_epochs = TRAINING_CONFIG["max_epochs"]
    accumulation_steps = TRAINING_CONFIG["accumulate_grad_batches"]
    max_train_batches = TRAINING_CONFIG["max_train_batches"]
    max_eval_batches = TRAINING_CONFIG["max_eval_batches"]
    eval_interval = TRAINING_CONFIG["eval_interval"]
    gradient_clip_val = TRAINING_CONFIG["gradient_clip_val"]

    print("\n>>> Starting Finetuning...")
    model.train()
    
    for epoch in range(num_epochs):
        print(f"\n=== Starting Epoch {epoch+1}/{num_epochs} ===")
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            if (i + 1) > max_train_batches:
                break
            
            # Prepare inputs
            train_inputs = {k: v for k, v in batch.items() if k != "ground_truth"}
            train_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in train_inputs.items()}
            
            # Forward pass
            try:
                outputs = model(**train_inputs, labels=batch["input_ids"].to(device))
                loss = outputs.loss / accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * accumulation_steps
                
                # Gradient accumulation step
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Evaluation and generation example
                if (i + 1) % eval_interval == 0:
                    print(f"\n[Training Step {i+1}] Loss: {loss.item() * accumulation_steps:.4f}")
                    
                    # Generate sample prediction
                    model.eval()
                    with torch.no_grad():
                        try:
                            gen_inputs = {k: v for k, v in batch.items() 
                                        if k in ["input_ids", "attention_mask", "pixel_values_videos"]}
                            gen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                        for k, v in gen_inputs.items()}
                            
                            generated_ids = model.generate(
                                **gen_inputs,
                                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                                do_sample=GENERATION_CONFIG["do_sample"],
                                temperature=GENERATION_CONFIG["temperature"],
                                top_p=GENERATION_CONFIG["top_p"],
                                pad_token_id=processor.tokenizer.pad_token_id,
                                eos_token_id=processor.tokenizer.eos_token_id
                            )
                            
                            prompt_length = batch["input_ids"].shape[1]
                            new_tokens = generated_ids[:, prompt_length:]
                            predictions = processor.batch_decode(
                                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                            )
                            
                            print("Sample Prediction:", predictions[0][:100] + "..." if len(predictions[0]) > 100 else predictions[0])
                            print("Ground Truth:", batch["ground_truth"][0])
                            
                        except Exception as e:
                            print(f"Generation error: {e}")
                    
                    model.train()
                    
            except Exception as e:
                print(f"Training error at batch {i}: {e}")
                continue
        
        # Calculate average loss
        avg_loss = epoch_loss / min(len(train_loader), max_train_batches)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Training Loss: {avg_loss:.4f}")
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Run evaluation
        if eval_loader is not None:
            print(">>> Starting Evaluation")
            evaluate_model(model, processor, eval_loader, device, max_eval_batches)
    
    print(">>> Training Complete!")
    return model

def evaluate_model(model, processor, eval_loader, device, max_eval_batches=None):
    """
    Evaluate the model on evaluation dataset.
    
    Args:
        model: The model to evaluate
        processor: The processor
        eval_loader: Evaluation data loader
        device: Device
        max_eval_batches: Maximum number of evaluation batches
    """
    model.eval()
    all_predictions = []
    eval_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluation"):
            if max_eval_batches and eval_batches >= max_eval_batches:
                break
            
            try:
                gen_inputs = {k: v for k, v in batch.items() 
                            if k in ["input_ids", "attention_mask", "pixel_values_videos"]}
                gen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in gen_inputs.items()}
                
                generated_ids = model.generate(
                    **gen_inputs,
                    max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                    do_sample=False,
                    num_beams=GENERATION_CONFIG.get("num_beams", 1),
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
                
                prompt_length = batch["input_ids"].shape[1]
                new_tokens = generated_ids[:, prompt_length:]
                predictions = processor.batch_decode(
                    new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                
                ground_truth = batch.get("ground_truth", ["N/A"])
                all_predictions.append((predictions, ground_truth))
                eval_batches += 1
                
            except Exception as e:
                print(f"Evaluation error: {e}")
                continue
    
    # Print sample predictions
    print(">>> Evaluation Results:")
    for i, (preds, gt) in enumerate(all_predictions[:5]):
        print(f"Sample {i+1}:")
        print("Prediction:", preds[0] if preds else "N/A")
        print("Ground Truth:", gt[0] if gt else "N/A")
        print("-" * 50)
    
    model.train()
    return all_predictions