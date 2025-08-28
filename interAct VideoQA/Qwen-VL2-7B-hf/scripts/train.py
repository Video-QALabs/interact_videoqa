import torch
from transformers import Trainer, TrainingArguments
from config import config_dict
from model import load_model, configure_lora
from dataset import VideoLlama3Dataset, video_llama3_collate_fn

def compute_metrics(eval_pred):
    return {}

def train():
    model, tokenizer, processor = load_model()
    model = configure_lora(model)
    
    csv_file = "../Videos/video_annotation.csv" 
    video_dir = "../Videos"      
    
    dataset = VideoLlama3Dataset(csv_file, video_dir, processor,
                                 num_frames=config_dict["num_frames"],
                                 target_size=config_dict.get("target_frame_size"))
    
    training_args = TrainingArguments(
        output_dir="./qwen2vl_outputs",
        num_train_epochs=config_dict["max_epochs"],
        per_device_train_batch_size=config_dict["batch_size"],
        learning_rate=config_dict["learning_rate"],
        fp16=True,
        logging_steps=1,
        save_steps=100,
        evaluation_strategy="no",
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=video_llama3_collate_fn,
        compute_metrics=compute_metrics
    )
    
    torch.cuda.empty_cache()
    trainer.train()


