#!/usr/bin/env python3
"""
InterVL3 Fine-tuning Script with LoRA Support
Supports video question-answering fine-tuning with eager execution
"""

import os
import json
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    TrainingArguments, Trainer,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from pathlib import Path
import logging
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoQADataset(torch.utils.data.Dataset):
    """Dataset class for video question-answering"""
    
    def __init__(self, jsonl_path: str, video_root: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.video_root = video_root
        self.max_length = max_length
        self.samples = []
        
        # Load JSONL data
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.samples.append(data)
        
        logger.info(f"Loaded {len(self.samples)} training samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract conversation
        conversations = sample['conversations']
        video_path = os.path.join(self.video_root, sample['video'])
        
        # Build conversation text
        conversation_text = ""
        for conv in conversations:
            if conv['from'] == 'human':
                conversation_text += f"Human: {conv['value']}\n"
            else:
                conversation_text += f"Assistant: {conv['value']}\n"
        
        # Tokenize
        encoding = self.tokenizer(
            conversation_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
            'video_path': video_path,
            'sample_id': sample.get('id', f'sample_{idx}')
        }

class InterVL3Trainer:
    """Main trainer class for InterVL3 fine-tuning"""
    
    def __init__(self, args):
        self.args = args
        self.setup_distributed()
        self.setup_model_and_tokenizer()
        self.setup_lora()
        self.setup_data()
        
    def setup_distributed(self):
        """Setup distributed training if available"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.args.rank = int(os.environ["RANK"])
            self.args.world_size = int(os.environ['WORLD_SIZE'])
            self.args.gpu = int(os.environ['LOCAL_RANK'])
            
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.args.gpu)
        else:
            self.args.rank = 0
            self.args.world_size = 1
            self.args.gpu = 0 if torch.cuda.is_available() else -1
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with eager execution"""
        logger.info(f"Loading model: {self.args.model_name_or_path}")
        
        # Load configuration
        config = AutoConfig.from_pretrained(
            self.args.model_name_or_path,
            trust_remote_code=True
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Ensure tokenizer has proper tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with eager execution
        self.model = AutoModel.from_pretrained(
            self.args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            attn_implementation="eager",
            trust_remote_code=True
        )
        
        # Freeze components as specified
        if self.args.freeze_backbone:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
            logger.info("Frozen vision backbone")
        
        if self.args.freeze_mlp:
            for param in self.model.mlp1.parameters():
                param.requires_grad = False
            logger.info("Frozen MLP layers")
            
        logger.info("Model loaded successfully with eager attention")
    
    def setup_lora(self):
        """Setup LoRA configuration"""
        if self.args.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias="none"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            logger.info("LoRA configuration applied")
    
    def setup_data(self):
        """Setup training data"""
        # Load meta configuration
        with open(self.args.meta_path, 'r') as f:
            meta_config = json.load(f)
        
        # Get first dataset configuration
        dataset_config = list(meta_config.values())[0]
        
        # Create dataset
        self.train_dataset = VideoQADataset(
            jsonl_path=dataset_config['annotation'],
            video_root=dataset_config['root'],
            tokenizer=self.tokenizer,
            max_length=self.args.max_seq_length
        )
        
        logger.info(f"Created training dataset with {len(self.train_dataset)} samples")
    
    def train(self):
        """Main training loop"""
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            overwrite_output_dir=self.args.overwrite_output_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            warmup_ratio=self.args.warmup_ratio,
            lr_scheduler_type=self.args.lr_scheduler_type,
            logging_steps=self.args.logging_steps,
            save_steps=self.args.save_steps,
            save_total_limit=self.args.save_total_limit,
            evaluation_strategy="no",
            bf16=self.args.bf16,
            gradient_checkpointing=self.args.grad_checkpoint,
            dataloader_num_workers=self.args.dataloader_num_workers,
            remove_unused_columns=False,
            report_to="tensorboard" if self.args.report_to else None,
            run_name=f"internvl3_lora_{self.args.run_name}" if self.args.run_name else None
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        if self.args.rank == 0:
            logger.info(f"Saving model to {self.args.output_dir}")
            trainer.save_model()
            self.tokenizer.save_pretrained(self.args.output_dir)

def parse_args():
    """Parse command line arguments - Official InterVL3 format"""
    parser = argparse.ArgumentParser(description="InterVL3 Fine-tuning with LoRA")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Path to pretrained model")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for model checkpoints")
    parser.add_argument("--meta_path", type=str, required=True,
                       help="Path to meta configuration JSON file")
    
    # Official InterVL3 arguments (fixed boolean parsing)
    parser.add_argument("--conv_style", type=str, default="Hermes-2",
                       help="Conversation style")
    parser.add_argument("--overwrite_output_dir", action="store_true", default=False,
                       help="Overwrite output directory")
    parser.add_argument("--force_image_size", type=int, default=448,
                       help="Force image size")
    parser.add_argument("--max_dynamic_patch", type=int, default=6,
                       help="Maximum dynamic patches")
    parser.add_argument("--down_sample_ratio", type=float, default=0.5,
                       help="Down sample ratio")
    parser.add_argument("--drop_path_rate", type=float, default=0.0,
                       help="Drop path rate")
    parser.add_argument("--vision_select_layer", type=int, default=-1,
                       help="Vision select layer")
    parser.add_argument("--ps_version", type=str, default="v2",
                       help="PS version")
    parser.add_argument("--dynamic_image_size", action="store_true", default=False,
                       help="Dynamic image size")
    parser.add_argument("--use_thumbnail", action="store_true", default=False,
                       help="Use thumbnail")
    parser.add_argument("--group_by_length", action="store_true", default=False,
                       help="Group by length")
    parser.add_argument("--do_train", action="store_true", default=False,
                       help="Do training")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05,
                       help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                       help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                       help="Learning rate scheduler type")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--evaluation_strategy", type=str, default="no",
                       help="Evaluation strategy")
    parser.add_argument("--save_strategy", type=str, default="steps",
                       help="Save strategy")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Freezing arguments (fixed boolean parsing)
    parser.add_argument("--freeze_llm", action="store_true", default=False,
                       help="Freeze LLM")
    parser.add_argument("--freeze_backbone", action="store_true", default=False,
                       help="Freeze vision backbone")
    parser.add_argument("--freeze_mlp", action="store_true", default=False, 
                       help="Freeze MLP layers")
    
    # Other arguments (fixed boolean parsing)
    parser.add_argument("--bf16", action="store_true", default=False,
                       help="Use bfloat16 precision")
    parser.add_argument("--grad_checkpoint", action="store_true", default=False,
                       help="Use gradient checkpointing")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--logging_steps", type=int, default=1,
                       help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=200,
                       help="Save steps")
    parser.add_argument("--save_total_limit", type=int, default=1,
                       help="Maximum number of checkpoints to keep")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                       help="Reporting tool")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Run name for logging")
    
    return parser.parse_args()

def train_with_lora(args):
    """Train with LoRA - called from main.py"""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = InterVL3Trainer(args)
    
    # Start training
    trainer.train()
    
    logger.info("Training completed successfully!")