import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return model, tokenizer, processor

def find_tunable_parts(model):
    tunable = []
    for name, module in model.named_modules():
        lower_name = name.lower()
        if "vision" in lower_name or "projector" in lower_name or "multi_projector" in lower_name:
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d)):
                tunable.append(name)
        elif "transformer" in lower_name or "decoder" in lower_name or "lm_head" in lower_name:
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                tunable.append(name)
    return list(set(tunable))

def configure_lora(model):
    tunable_parts = find_tunable_parts(model)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=4,
        lora_dropout=0.1,
        target_modules=tunable_parts,
        init_lora_weights="gaussian",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model
