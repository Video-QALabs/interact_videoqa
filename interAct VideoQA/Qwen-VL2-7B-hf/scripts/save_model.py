import os
import torch
import json
from peft import get_peft_model_state_dict
from scripts.config import DATA_PATHS

def save_model(model, processor, save_dir=None):
    """
    Save the trained model and processor.
    
    Args:
        model: The trained model (with LoRA adapters)
        processor: The processor
        save_dir: Directory to save the model (optional)
    """
    if save_dir is None:
        save_dir = DATA_PATHS["save_dir"]
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        print(f"Saving model to: {save_dir}")
        
        # Save the model (this will save LoRA adapters if it's a PEFT model)
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(save_dir)
            print("✓ Model (with LoRA adapters) saved successfully")
        else:
            # Fallback: save state dict
            torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
            print("✓ Model state dict saved successfully")
        
        # Save the processor
        processor.save_pretrained(save_dir)
        print("✓ Processor saved successfully")
        
        # Save additional training info
        save_info = {
            "model_type": "llava-next-video-lora",
            "base_model": "llava-hf/LLaVA-NeXT-Video-7B-hf",
            "training_completed": True,
            "peft_enabled": hasattr(model, 'peft_config'),
            "model_size": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        with open(os.path.join(save_dir, "training_info.json"), "w") as f:
            json.dump(save_info, f, indent=2)
        print("✓ Training info saved")
        
        # Save model configuration if available
        if hasattr(model, 'config'):
            model.config.save_pretrained(save_dir)
            print("✓ Model config saved")
        
        print(f"\n✅ Model successfully saved to: {save_dir}")
        print("📁 Files saved:")
        for file in sorted(os.listdir(save_dir)):
            file_path = os.path.join(save_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                size_str = format_file_size(size)
                print(f"   - {file} ({size_str})")
            
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        raise e

def load_model(model_class, processor_class, save_dir=None, device_map="auto"):
    """
    Load a saved model and processor.
    
    Args:
        model_class: The model class (e.g., LlavaNextVideoForConditionalGeneration)
        processor_class: The processor class (e.g., LlavaNextVideoProcessor)
        save_dir: Directory containing the saved model
        device_map: Device mapping strategy
    
    Returns:
        tuple: (model, processor)
    """
    if save_dir is None:
        save_dir = DATA_PATHS["save_dir"]
    
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"Model directory not found: {save_dir}")
    
    try:
        print(f"📂 Loading model from: {save_dir}")
        
        # Check what files are available
        files = os.listdir(save_dir)
        print(f"📄 Found files: {files}")
        
        # Load processor
        processor = processor_class.from_pretrained(save_dir)
        print("✓ Processor loaded successfully")
        
        # Load model
        model = model_class.from_pretrained(
            save_dir,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )
        print("✓ Model loaded successfully")
        
        # Load training info if available
        info_path = os.path.join(save_dir, "training_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                info = json.load(f)
            print(f"📊 Model info: {info.get('model_type', 'Unknown')}")
            if info.get('peft_enabled'):
                print("🔧 LoRA adapters detected and loaded")
        
        return model, processor
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

def save_checkpoint(model, processor, optimizer, epoch, loss, checkpoint_dir=None):
    """
    Save a training checkpoint.
    
    Args:
        model: Current model state
        processor: Processor
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        checkpoint_dir: Directory to save checkpoint
    """
    if checkpoint_dir is None:
        checkpoint_dir = DATA_PATHS["checkpoint_dir"]
    
    # Create checkpoint directory with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_{timestamp}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    try:
        print(f"💾 Saving checkpoint to: {checkpoint_path}")
        
        # Save model
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(checkpoint_path)
        else:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, "pytorch_model.bin"))
        
        # Save processor
        processor.save_pretrained(checkpoint_path)
        
        # Save optimizer and training state
        checkpoint_state = {
            "epoch": epoch,
            "loss": loss,
            "optimizer_state_dict": optimizer.state_dict(),
            "timestamp": timestamp,
            "model_type": "llava-next-video-checkpoint"
        }
        
        torch.save(checkpoint_state, os.path.join(checkpoint_path, "training_state.pt"))
        
        # Save checkpoint info
        checkpoint_info = {
            "epoch": epoch,
            "loss": float(loss),
            "timestamp": timestamp,
            "checkpoint_path": checkpoint_path
        }
        
        with open(os.path.join(checkpoint_path, "checkpoint_info.json"), "w") as f:
            json.dump(checkpoint_info, f, indent=2)
        
        print(f"✅ Checkpoint saved successfully")
        print(f"📊 Epoch: {epoch}, Loss: {loss:.4f}")
        
        return checkpoint_path
        
    except Exception as e:
        print(f"❌ Error saving checkpoint: {e}")
        raise e

def load_checkpoint(checkpoint_path, model, processor, optimizer=None):
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        model: Model to load state into
        processor: Processor to load state into  
        optimizer: Optimizer to load state into (optional)
    
    Returns:
        dict: Training state information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        print(f"📂 Loading checkpoint from: {checkpoint_path}")
        
        # Load training state
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        if os.path.exists(state_path):
            checkpoint_state = torch.load(state_path, map_location="cpu")
            
            if optimizer and "optimizer_state_dict" in checkpoint_state:
                optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
                print("✓ Optimizer state loaded")
            
            epoch = checkpoint_state.get('epoch', 0)
            loss = checkpoint_state.get('loss', 0.0)
            
            print(f"✅ Checkpoint loaded successfully")
            print(f"📊 Resuming from epoch {epoch}, loss: {loss:.4f}")
            
            return checkpoint_state
        else:
            print("⚠️  No training state found, loading model only")
            return {"epoch": 0, "loss": 0.0}
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return {"epoch": 0, "loss": 0.0}

def list_checkpoints(checkpoint_dir=None):
    """
    List available checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        list: List of checkpoint information
    """
    if checkpoint_dir is None:
        checkpoint_dir = DATA_PATHS["checkpoint_dir"]
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return []
    
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path) and item.startswith("checkpoint_"):
            info_path = os.path.join(item_path, "checkpoint_info.json")
            if os.path.exists(info_path):
                try:
                    with open(info_path, "r") as f:
                        info = json.load(f)
                    info["path"] = item_path
                    checkpoints.append(info)
                except:
                    # Fallback info
                    checkpoints.append({
                        "path": item_path,
                        "epoch": "unknown",
                        "loss": "unknown",
                        "timestamp": "unknown"
                    })
    
    # Sort by epoch
    checkpoints.sort(key=lambda x: x.get("epoch", 0), reverse=True)
    
    print(f"📋 Found {len(checkpoints)} checkpoints:")
    for i, cp in enumerate(checkpoints):
        print(f"   {i+1}. Epoch {cp['epoch']}, Loss: {cp['loss']}, Path: {cp['path']}")
    
    return checkpoints

def cleanup_old_checkpoints(checkpoint_dir=None, keep_last=3):
    """
    Clean up old checkpoints, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of recent checkpoints to keep
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if len(checkpoints) <= keep_last:
        print(f"💾 Only {len(checkpoints)} checkpoints found, no cleanup needed")
        return
    
    # Remove old checkpoints
    to_remove = checkpoints[keep_last:]
    
    print(f"🧹 Cleaning up {len(to_remove)} old checkpoints...")
    for cp in to_remove:
        try:
            import shutil
            shutil.rmtree(cp["path"])
            print(f"   ✓ Removed: {cp['path']}")
        except Exception as e:
            print(f"   ❌ Failed to remove {cp['path']}: {e}")
    
    print(f"✅ Cleanup complete, kept {keep_last} most recent checkpoints")

def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def get_model_info(save_dir=None):
    """
    Get information about a saved model.
    
    Args:
        save_dir: Directory containing the saved model
    
    Returns:
        dict: Model information
    """
    if save_dir is None:
        save_dir = DATA_PATHS["save_dir"]
    
    if not os.path.exists(save_dir):
        return {"error": f"Model directory not found: {save_dir}"}
    
    info = {"path": save_dir, "files": []}
    
    # List files with sizes
    for file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            info["files"].append({
                "name": file,
                "size": size,
                "size_formatted": format_file_size(size)
            })
    
    # Load training info if available
    info_path = os.path.join(save_dir, "training_info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                training_info = json.load(f)
            info.update(training_info)
        except Exception as e:
            info["training_info_error"] = str(e)
    
    return info