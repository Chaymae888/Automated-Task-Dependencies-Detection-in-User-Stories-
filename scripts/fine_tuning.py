# Fixed Fine-tuning Script Compatible with Transformers 4.53.0
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
import re

# Better model choice for instruction following
MODEL_NAME = "microsoft/DialoGPT-medium"
OUTPUT_DIR = "./task-extractor-v2"
DATASET_PATH = "dataset.json"

def create_better_prompt_format(user_story, tasks):
    """Create a more structured prompt format"""
    # Format tasks as a numbered list
    if isinstance(tasks, list):
        task_list = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
    else:
        # If tasks is a string, try to split it
        task_items = [t.strip() for t in tasks.split(',') if t.strip()]
        task_list = "\n".join([f"{i+1}. {task}" for i, task in enumerate(task_items)])
    
    return f"""### Instruction:
Extract specific development tasks from the following user story.

### User Story:
{user_story}

### Tasks:
{task_list}

### End"""

def load_model_and_tokenizer():
    """Load model with better configuration"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Add special tokens
    special_tokens = {
        "pad_token": "<pad>",
        "eos_token": "</s>",
        "bos_token": "<s>",
    }
    
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Resize token embeddings if we added tokens
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    # Better LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Increased rank
        lora_alpha=32,  # Higher alpha for stronger adaptation
        lora_dropout=0.05,  # Lower dropout
        target_modules=["c_attn", "c_proj"],  # Target more modules
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def prepare_improved_dataset(dataset_path, tokenizer, max_length=512):
    """Prepare dataset with better formatting"""
    print("Loading dataset...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} training examples")
    
    # Create better formatted training data
    formatted_data = []
    for item in raw_data:
        # Use the structured prompt format
        formatted_text = create_better_prompt_format(item['input'], item['output'])
        formatted_data.append({"text": formatted_text})
    
    # Print a sample to verify format
    print("\nSample training format:")
    print(formatted_data[0]['text'][:300] + "...")
    
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
        
        # Mask padding tokens in labels
        tokens["labels"][tokens["labels"] == tokenizer.pad_token_id] = -100
        
        return tokens
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    return tokenized_dataset

def get_compatible_training_arguments():
    """Training arguments compatible with Transformers 4.53.0"""
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,  # Reduced for faster testing
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=5e-5,  # Slightly lower learning rate
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",  # Fixed: was evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        lr_scheduler_type="cosine",
        prediction_loss_only=True,
        # Remove problematic arguments for older versions
        # report_to="none",  # Disable wandb/tensorboard
    )

def create_evaluation_dataset(dataset, split_ratio=0.1):
    """Split dataset for evaluation"""
    dataset_size = len(dataset)
    eval_size = int(dataset_size * split_ratio)
    
    # Simple split
    eval_dataset = dataset.select(range(eval_size))
    train_dataset = dataset.select(range(eval_size, dataset_size))
    
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    return train_dataset, eval_dataset

def test_model_with_multiple_examples(model, tokenizer):
    """Test with multiple examples"""
    test_stories = [
        "As a user, I want to find nearby recycling centers on a map",
        "As a developer, I want to implement user authentication, so that users can securely access their accounts",
        "As an admin, I want to manage user permissions, so that I can control access to different features"
    ]
    
    print("\n=== Testing Model ===")
    model.eval()  # Set to evaluation mode
    
    for i, story in enumerate(test_stories, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {story}")
        
        # Use the same format as training
        prompt = f"""### Instruction:
Extract specific development tasks from the following user story.

### User Story:
{story}

### Tasks:
"""
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        generated_part = response[len(prompt):].strip()
        
        # Clean up the output
        if "### End" in generated_part:
            generated_part = generated_part.split("### End")[0].strip()
        
        print(f"Generated Tasks: {generated_part}")

def main():
    """Main training function"""
    # Check dataset
    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found!")
        print("Please make sure you have a dataset.json file in the current directory.")
        return
    
    # Load model and tokenizer
    print("=== Loading Model ===")
    try:
        model, tokenizer = load_model_and_tokenizer()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Prepare dataset
    print("\n=== Preparing Dataset ===")
    try:
        dataset = prepare_improved_dataset(DATASET_PATH, tokenizer)
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        return
    
    # Split for evaluation
    train_dataset, eval_dataset = create_evaluation_dataset(dataset)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = get_compatible_training_arguments()
    
    # Create trainer with evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
    )
    
    # Start training
    print("\n=== Starting Training ===")
    try:
        trainer.train()
        print("✓ Training completed successfully!")
        
        # Print final training metrics
        train_result = trainer.state.log_history
        if train_result:
            final_loss = train_result[-1].get('train_loss', 'N/A')
            print(f"Final training loss: {final_loss}")
            
    except Exception as e:
        print(f"✗ Training failed: {e}")
        print("This might be due to memory constraints or other issues.")
        return
    
    # Save the model
    print("\n=== Saving Model ===")
    try:
        trainer.save_model()
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✓ Model saved to {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return
    
    # Test the model
    try:
        test_model_with_multiple_examples(model, tokenizer)
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("-" * 50)
    
    main()