# =============================================================================
# 3. Model & Processor Setup with QLora / 4-bit Quantization
# =============================================================================
model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
processor = LlavaNextVideoProcessor.from_pretrained(model_id)
processor.tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
)

model.config.use_cache = False

def find_vision_target_modules(model):
    target_names = []
    for name, module in model.named_modules():
        if "multimodal_projector" in name and isinstance(module, torch.nn.Linear):
            target_names.append(name)
    return list(set(target_names))

def find_language_target_modules(model):
    target_names = []
    for name, module in model.named_modules():
        if "language_model.model.layers" in name and isinstance(module, torch.nn.Linear):
            if any(sub in name for sub in ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]):
                target_names.append(name)
    return list(set(target_names))

# Combine target modules for both vision and language.
tunable_parts = find_vision_target_modules(model) + find_language_target_modules(model)
