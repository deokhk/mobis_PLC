import os
import pdb
import json
import wandb
import torch
import argparse 

from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from accelerate import PartialState
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, EarlyStoppingCallback


def main(args):
    # os.environ['CUDA_LAUNCH_BLOCKING']="1"
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
		
		## 데이터세트 준비
    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_data, split="train")
		
    num_gpus = torch.cuda.device_count()
    grad_accum_steps = args.batch_size // args.per_device_train_batch_size // num_gpus
		
		## LoRA
    # lora_r = args.lora_r
    # peft_config = LoraConfig(
    #     lora_alpha=lora_r*2,
    #     lora_dropout=0.05,
    #     r=lora_r,
    #     bias="none",
    #     task_type="CAUSAL_LM"
    # )
		
		
		## 학습 파라미터
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        dataloader_drop_last=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        run_name=args.wandb_run_name,
        report_to="wandb",
        save_only_model=True,
    )
		
		
		## 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map={'':device_string},
    )
    
    
    ## Trainer 로드
    #tokenizer.padding_side = "left"
    model.config.use_cache = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # peft_config=peft_config,
        max_seq_length=4096,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=1)],
    )
		
		
		## 학습
    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with SFTTrainer")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--train_data", type=str, required=True, help="Path to the training data file")
    parser.add_argument("--eval_data", type=str, required=True, help="Path to the evaluation data file")
    
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (including gradient accumulation, multi-gpu training)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device during training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size per device during evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="The initial learning rate for Adam")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="The scheduler type to use", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay if we apply some")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Linear warmup over warmup_ratio fraction of total steps")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Name of the W&B run")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier to load from huggingface.co/models")

    args = parser.parse_args()

    main(args)
    args = parser.parse_args()

    main(args)
