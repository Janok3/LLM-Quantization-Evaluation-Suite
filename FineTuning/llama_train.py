import os
import torch
import sys
import argparse
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

# --- Configuration ---
MODEL_ID = "Qwen/Qwen3.5-4B"
DATASET_NAME = "mlabonne/guanaco-llama2-1k" 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./results_falcon")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    
    # 1. Load Tokenizer
    print(f"Loading Tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # FIX 1: Set length here to avoid SFTTrainer arg issues
    tokenizer.model_max_length = 512 

    # 2. Load Base Model
    print(f"Loading Model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
        # FIX 2: Use float32 for CPU stability since GPU driver is broken
        dtype=torch.float32, 
    )
    
    # 3. PEFT Config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=16, 
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] 
    )

    # 4. Load Dataset
    dataset = load_dataset(DATASET_NAME, split="train")
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # 5. Formatting Function
    def formatting_prompts_func(example):
        output_texts = []
        for text in example['text']:
            output_texts.append(text)
        return output_texts

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        save_steps=50,
        logging_steps=10,
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        # FIX 3: New eval strategy name
        eval_strategy="steps", 
        eval_steps=25,
        save_strategy="steps",
        load_best_model_at_end=True,
        use_cpu=False,
        gradient_checkpointing=False, 
    )

    # 7. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        # FIX 4: processing_class instead of tokenizer
        processing_class=tokenizer,  
        args=training_args,
        # FIX 5: 'packing' argument removed entirely
    )

    # 8. Train
    print("Starting Training...")
    trainer.train()

    # 9. Save
    print("Saving model...")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_adapter"))

if __name__ == "__main__":
    main()