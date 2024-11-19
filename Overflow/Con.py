import requests

def main():
    api_url = "http://10.100.0.34:5000/generate"

    instruction = "You're a helpful assistant. Return only the answer to the question."
    question = "What is the tallest building in the world?"

    prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {instruction}

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        {question}
        <|end_of_text|>
        """

    payload = {
        "prompt": prompt,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.1,
        "max_length": 512,
        "max_new_tokens": 100,
    }

    try:
        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            print(response.json().get("response", "No response key in JSON"))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    main()
