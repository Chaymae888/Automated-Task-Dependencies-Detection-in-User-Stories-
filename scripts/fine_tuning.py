# Simple Compatible Fine-tuning Script
# Works with older versions of transformers

import torch
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os

# Configuration
MODEL_NAME = "microsoft/DialoGPT-small"
OUTPUT_DIR = "./user-stories-task-extractor"
DATASET_PATH = "dataset.json"

def load_model_and_tokenizer():
    """Load model and tokenizer with CPU compatibility"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Simple LoRA config that works with DialoGPT
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Smaller rank for stability
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn"],  # Only target attention layers
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_dataset(dataset_path, tokenizer, max_length=256):
    """Prepare dataset with shorter sequences for CPU training"""
    print("Loading dataset...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} training examples")
    
    # Simple format
    formatted_data = []
    for item in raw_data:
        # Keep it simple - just input -> output
        text = f"Extract tasks from: {item['input']}\nTasks: {item['output']}<|endoftext|>"
        formatted_data.append({"text": text})
    
    dataset = Dataset.from_list(formatted_data)
    
    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    return tokenized_dataset

def get_simple_training_arguments():
    """Minimal training arguments that work with older transformers"""
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=2,  # Reduced for CPU
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=3e-5,
        logging_steps=5,
        save_steps=50,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        # Avoid problematic parameters
        prediction_loss_only=True,
    )

def main():
    """Simple training function"""
    # Check dataset
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found!")
        return
    
    # Load model and tokenizer
    print("=== Loading Model ===")
    model, tokenizer = load_model_and_tokenizer()
    
    # Prepare dataset
    print("\n=== Preparing Dataset ===")
    dataset = prepare_dataset(DATASET_PATH, tokenizer)
    
    print(f"Total samples: {len(dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = get_simple_training_arguments()
    
    # Create trainer (no evaluation to avoid compatibility issues)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("\n=== Starting Training ===")
    try:
        trainer.train()
        print("✓ Training completed successfully!")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return
    
    # Save the model
    print("\n=== Saving Model ===")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ Model saved to {OUTPUT_DIR}")
    
    # Test the model
    print("\n=== Testing Model ===")
    test_user_story = "As a user, I want to find nearby recycling centers on a map"
    
    # Simple generation test
    prompt = f"Extract tasks from: {test_user_story}\nTasks:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Input: {test_user_story}")
    print(f"Output: {response}")

if __name__ == "__main__":
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("-" * 50)
    
    main()