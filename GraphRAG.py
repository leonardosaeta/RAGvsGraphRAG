from neo4j import GraphDatabase
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize Neo4j connection
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "your_neo4j_password"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Load local model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # or another model available locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define function to retrieve related content from Neo4j
def get_related_content(query):
    with driver.session() as session:
        results = session.run(
            """
            MATCH (n:Guide {name: 'Installation Guide'})-[:RELATED_TO|PREREQUISITE_FOR*]->(related)
            RETURN n.content AS main_content, collect(related.content) AS related_content
            """
        )
        record = results.single()
        if record:
            main_content = record["main_content"]
            related_content = record["related_content"]
            return main_content, related_content
        return None, None

# Function to generate answer using local LLM
def generate_response(main_content, related_content, user_query):
    context = f"Main Content: {main_content}\n\nRelated Information:\n"
    context += "\n".join(related_content)
    prompt = f"User Query: {user_query}\n\n{context}\n\nAnswer the query based on the information provided above."

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    # Generate a response
    outputs = model.generate(inputs.input_ids, max_length=150, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Main function
def main():
    # User query
    user_query = "How do I install the software?"

    # Retrieve relevant information from Neo4j
    main_content, related_content = get_related_content(user_query)

    if main_content and related_content:
        # Generate response using local LLM
        answer = generate_response(main_content, related_content, user_query)
        print("Response:", answer)
    else:
        print("No relevant information found.")

# Run the main function
if __name__ == "__main__":
    main()
