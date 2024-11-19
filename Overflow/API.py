import re
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import os
import json

app = Flask(__name__)

# Initialize the model and tokenizer globally
model = None
tokenizer = None

def init():
    global model
    global tokenizer

    tempo_ini = time.time()

    # Configuration for quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4"
    )

    # Load the language model
    model_path = "Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto"
    )

    print("Init complete")
    print("Tempo de inicialização = {}".format(round(time.time() - tempo_ini, 3)))

def str2bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1')
    return bool(value)

def generate_prompt_answer(prompt,
                           repetition_penalty=1.1,
                           do_sample=True,
                           temperature=0.1,
                           top_p=0.1,
                           max_length=756,
                           max_new_tokens=1024):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs['input_ids']
    input_length = input_ids.shape[1]

    # Set generation parameters
    generation_kwargs = dict(
        input_ids=input_ids,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_length=max_length,
        max_new_tokens=max_new_tokens
    )

    # Generate text
    output = model.generate(**generation_kwargs)

    # Decode only the newly generated tokens
    generated_tokens = output[0, input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text.strip()


@app.route('/generate', methods=['POST'])
def generate_answer():
    data = request.json
    prompt = data.get('prompt', None)

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Get generation parameters with default values
    repetition_penalty = float(data.get('repetition_penalty', 1.1))
    do_sample = str2bool(data.get('do_sample', True))
    temperature = float(data.get('temperature', 0.1))
    top_p = float(data.get('top_p', 0.1))
    max_length = int(data.get('max_length', 756))
    max_new_tokens = int(data.get('max_new_tokens', 1024))

    # Generate the full response
    response = generate_prompt_answer(
        prompt,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    
    return jsonify({"response": response})

init()
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
