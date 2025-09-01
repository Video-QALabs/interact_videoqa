from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
import torch
import os
from peft import LoraConfig, get_peft_model
from scripts.config import MODEL_CONFIG, LORA_CONFIG

def setup_qwen_model(checkpoint_path=None, cache_dir=None):
    """
    Setup the Qwen2.5-VL model with LoRA configuration.
    
    Args:
        checkpoint_path (str, optional): Path to load model checkpoint
        cache_dir (str, optional): Custom cache directory
    
    Returns:
        tuple: (model, processor)
    """
    model_id = MODEL_CONFIG["model_id"]
    
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
        print(f"Using custom cache directory: {cache_dir}")
    
    # Setup processor and tokenizer
    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Configure tokenizer
    tokenizer.padding_side = "left"

    
    if checkpoint_path:
        # Load from checkpoint
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        # Setup quantization config for training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load base model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Disable cache for training
        model.config.use_cache = False
        
        # Setup LoRA configuration for Qwen2.5-VL
        target_modules = find_qwen_target_modules(model)
        lora_config = LoraConfig(
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            target_modules=target_modules,
            lora_dropout=LORA_CONFIG["lora_dropout"],
            bias=LORA_CONFIG["bias"],
            task_type=LORA_CONFIG["task_type"]
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
    
    return model, processor, tokenizer

def find_qwen_target_modules(model):
    """Find all modules that can be trained with LoRA in Qwen2.5-VL."""
    target_modules = set()
    
    # Vision modules - visual projection layers
    for name, module in model.named_modules():
        if "visual" in name and isinstance(module, torch.nn.Linear):
            if any(sub in name for sub in ["proj", "fc", "linear"]):
                target_modules.add(name.split('.')[-1])  # Get module name only
    
    # Language model modules - Qwen2.5 architecture
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(sub in name for sub in ["q_proj", "k_proj", "v_proj", "o_proj",
                                         "up_proj", "down_proj", "gate_proj",
                                         "lm_head"]):
                target_modules.add(name.split('.')[-1])  # Get module name only
    
    # Convert to list and remove duplicates
    target_modules = list(set(target_modules))
    
    # If no modules found, use common defaults for Qwen
    if not target_modules:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                         "up_proj", "down_proj", "gate_proj"]
    
    print(f"Found {len(target_modules)} target modules for LoRA:")
    for module in target_modules:
        print(f"  - {module}")
    
    return target_modules