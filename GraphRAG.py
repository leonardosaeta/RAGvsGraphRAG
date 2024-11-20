import requests
# from py2neo import Graph, Node, Relationship
# import pdfplumber
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='Auth/config.env') 

# Neo4j configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Llama 3.1 API configuration
LLAMA_API_URL = "http://10.100.0.34:5000/generate"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text_chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_chunks.append(text.strip())
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text_chunks

def analyze_text_with_llama(text):
    instruction = "Extract entities and relationships for a knowledge graph."
    prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {instruction}

        <|eot_id|><|start_header_id|>user<|end_header_id|>
        {text}
        <|end_of_text|>
    """
    payload = {
        "prompt": prompt,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_length": 512,
        "max_new_tokens": 100
    }
    try:
        response = requests.post(LLAMA_API_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("response", "No response key in JSON")
        else:
            print(f"Llama API Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Request to Llama failed: {e}")
    return None

# Function to store data in Neo4j
def store_in_neo4j(entities, relationships):
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        for entity in entities:
            node = Node("Entity", name=entity)
            graph.merge(node, "Entity", "name")

        for relationship in relationships:
            node1 = graph.nodes.match("Entity", name=relationship["from"]).first()
            node2 = graph.nodes.match("Entity", name=relationship["to"]).first()
            if node1 and node2:
                rel = Relationship(node1, relationship["type"], node2)
                graph.merge(rel)
    except Exception as e:
        print(f"Error storing data in Neo4j: {e}")

# Main function
def main(pdf_path):
    text_chunks = extract_text_from_pdf(pdf_path)
    for chunk in text_chunks:
        llama_response = analyze_text_with_llama(chunk)
        if llama_response:
            print(f"Llama Response: {llama_response}")
            # Simulated processing of llama_response into entities/relationships
            entities = ["Entity1", "Entity2"]  # Replace with parsed entities
            relationships = [
                {"from": "Entity1", "to": "Entity2", "type": "RELATED_TO"}
            ]  # Replace with parsed relationships
            store_in_neo4j(entities, relationships)

if __name__ == "__main__":
    pdf_file_path = "path_to_your_pdf_document.pdf"
    main(pdf_file_path)
