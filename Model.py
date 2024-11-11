from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from threading import Thread
import torch

# Load model with 4-bit quantization
model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define quantization config properly
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load model with quantization config
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

def generate_with_streaming(prompt, max_new_tokens=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Setup streamer
    streamer = TextIteratorStreamer(tokenizer)
    
    # Generation configuration
    gen_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "top_p": 0.9,
    }
    
    # Create generation thread
    generation_kwargs = dict(**gen_config, **inputs)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream the output
    print("Generating response...")
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        print(f"\r{generated_text}", end="", flush=True)
    print("\nDone!")

# Example usage
input_text = "Qual o maior pais do mundo?"
generate_with_streaming(input_text, max_new_tokens=150)

