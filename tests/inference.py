import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
MODEL_NAME = "microsoft/DialoGPT-small"  # Original base model
ADAPTER_PATH = "./user-stories-task-extractor"  

def load_model_for_inference():
    """Load base model with fine-tuned adapter weights"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float32
    )
    
    print("Loading adapter weights...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload()  
    
    return model, tokenizer

def generate_response(model, tokenizer, user_story):
    prompt = f"Extract tasks from: {user_story}\nTasks:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    generation_config = {
        "max_new_tokens": 150,
        "temperature": 0.3,
        "do_sample": False,  
        "num_beams": 3,
        "early_stopping": True,
        "repetition_penalty": 1.5
    }
    
    outputs = model.generate(**inputs, **generation_config)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the tasks portion
    tasks = response.split("EXTRACTED TASKS:")[-1].strip()
    return tasks

if __name__ == "__main__":
    model, tokenizer = load_model_for_inference()
    
    while True:
        user_input = input("\nEnter a user story (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
            
        response = generate_response(model, tokenizer, user_input)
        print("\nModel Output:")
        print(response)