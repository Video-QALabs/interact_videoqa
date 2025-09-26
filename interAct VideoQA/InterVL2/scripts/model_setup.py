import torch
from modelscope import AutoTokenizer,AutoModel
import math
from modelscope import AutoTokenizer,AutoModel,AutoConfig
from scripts.config import MODEL_NAME,MODEL_PATH,CACHE_DIR
import os
os.environ['MODELSCOPE_CACHE'] = '/scratch/jnolas77/VideoQA/interact_videoqa/interAct VideoQA/InterVL2/scripts/cache_dir'
def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

def setup_model(model_name=MODEL_NAME, model_path=MODEL_PATH, cache_dir=CACHE_DIR):
    """Setup and load the InternVL3 model with eager execution"""
    device_map = split_model(model_path)

    path = "OpenGVLab/InternVL3-38B"
    
    # Set up model configuration for eager execution
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    
    # Load model with eager attention implementation
    model = AutoModel.from_pretrained(
        path,
        config=config,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        attn_implementation="eager",
        trust_remote_code=True,
        cache_dir=cache_dir,
        device_map=device_map
    ).eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        path, 
        trust_remote_code=True, 
        use_fast=False,
        cache_dir=cache_dir
    )
    
    # Ensure tokenizer has proper padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully with eager attention")
    print(f"Model device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'Not available'}")
    
    return model, tokenizer