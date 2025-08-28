from transformers import LlavaNextVideoProcessor, BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration
import torch
from peft import LoraConfig, get_peft_model
from config import MODEL_CONFIG, LORA_CONFIG

def setup_llava_model(checkpoint_path=None):
    """
    Setup the LLaVA-Next-Video model with LoRA configuration.
    
    Args:
        checkpoint_path (str, optional): Path to load model checkpoint
    
    Returns:
        tuple: (model, processor)
    """
    model_id = MODEL_CONFIG["model_id"]
    
    # Setup processor
    processor = LlavaNextVideoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    
    # Add pad token if it doesn't exist
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    
    if checkpoint_path:
        # Load from checkpoint
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
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
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Disable cache for training
        model.config.use_cache = False
        
        # Setup LoRA configuration
        target_modules = find_all_target_modules(model)
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
    
    return model, processor

def find_all_target_modules(model):
    """Find all modules that can be trained with LoRA."""
    target_modules = set()
    
    # Vision modules - multimodal projector
    for name, module in model.named_modules():
        if "multi_modal_projector" in name and isinstance(module, torch.nn.Linear):
            target_modules.add(name)
    
    # Language model modules
    for name, module in model.named_modules():
        if "language_model.model.layers" in name and isinstance(module, torch.nn.Linear):
            if any(sub in name for sub in ["q_proj", "k_proj", "v_proj", "o_proj",
                                         "up_proj", "down_proj", "gate_proj"]):
                target_modules.add(name)
    
    # Convert to list and print for debugging
    target_modules = list(target_modules)
    print(f"Found {len(target_modules)} target modules for LoRA:")
    for module in target_modules[:5]:  # Print first 5
        print(f"  - {module}")
    if len(target_modules) > 5:
        print(f"  ... and {len(target_modules) - 5} more")
    
    return target_modules