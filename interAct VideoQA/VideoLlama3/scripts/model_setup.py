import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from scripts.config import MODEL_CONFIG, LORA_CONFIG, HARDWARE_CONFIG

def load_videollama3_model():
    """Load VideoLLaMA3 model and processor."""
    print(f"Loading VideoLLaMA3 model: {MODEL_CONFIG['model_id']}")
    
    # Load model with proper configuration
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG["model_id"],
        trust_remote_code=True,
        device_map={"": HARDWARE_CONFIG["device"]},
        torch_dtype=torch.bfloat16 if MODEL_CONFIG["torch_dtype"] == "bfloat16" else torch.float16,
        attn_implementation=MODEL_CONFIG.get("attn_implementation", "flash_attention_2"),
        cache_dir=MODEL_CONFIG.get("cache_dir"),
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        MODEL_CONFIG["model_id"],
        trust_remote_code=True,
        cache_dir=MODEL_CONFIG.get("cache_dir"),
    )
    
    print("VideoLLaMA3 model and processor loaded successfully!")
    return model, processor

def find_target_modules(model):
    """Find target modules for LoRA fine-tuning in VideoLLaMA3."""
    target_modules = set()
    
    for name, module in model.named_modules():
        # Target language model layers
        if any(target in name for target in ["q_proj", "v_proj", "k_proj", "o_proj", 
                                            "gate_proj", "up_proj", "down_proj"]):
            if isinstance(module, torch.nn.Linear):
                target_modules.add(name.split(".")[-1])
    
    # Use predefined target modules if automatic detection fails
    if not target_modules:
        target_modules = set(LORA_CONFIG["target_modules"])
    
    print(f"Target modules for LoRA: {list(target_modules)}")
    return list(target_modules)

def configure_lora_for_videollama3(model):
    """Configure LoRA for VideoLLaMA3 fine-tuning."""
    print("Configuring LoRA for VideoLLaMA3...")
    
    # Find target modules
    target_modules = find_target_modules(model)
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=LORA_CONFIG["task_type"],
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

def setup_videollama3_model(use_lora=True):
    """Complete setup for VideoLLaMA3 model."""
    # Load base model and processor
    model, processor = load_videollama3_model()
    
    # Configure LoRA if requested
    if use_lora:
        model = configure_lora_for_videollama3(model)
    
    return model, processor

# Utility function for inference setup
def load_model_for_inference(model_path=None):
    """Load model specifically for inference."""
    if model_path is None:
        model_path = MODEL_CONFIG["model_id"]
    
    print(f"Loading model for inference: {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": HARDWARE_CONFIG["device"]},
        torch_dtype=torch.bfloat16,
        attn_implementation=MODEL_CONFIG.get("attn_implementation", "flash_attention_2"),
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    return model, processor