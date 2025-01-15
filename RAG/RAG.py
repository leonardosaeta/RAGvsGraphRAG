import os
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
import textwrap

load_dotenv()

TEXT_FILES_DIR = os.getenv('TEXT_FILES_PATH')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if TEXT_FILES_DIR is None or OPENAI_API_KEY is None:
    raise ValueError("Required environment variables not found")

DATA_DIR = "./chroma_db"

openai_client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=DATA_DIR)

def read_and_chunk_text(file_path, chunk_size=1000, overlap=100):
    """Read a text file and break it into overlapping chunks."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                last_period = text[end-overlap:end].rfind('.')
                if last_period != -1:
                    end = end - overlap + last_period + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        
        return chunks
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return []

def get_or_create_collection(client, collection_name):
    """Get or create a collection."""
    try:
        return client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        print(f"Error creating collection: {str(e)}")
        raise

def add_documents_to_collection(collection, model, text_chunks, source_file):
    """Generate embeddings for text chunks and add them to the collection."""
    try:
        embeddings = model.encode(text_chunks)
        
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(text_chunks):
            chunk_id = f"{os.path.basename(source_file)}_{i}"
            documents.append(chunk)
            metadatas.append({"source": source_file, "chunk_index": i})
            ids.append(chunk_id)

        collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Added {len(text_chunks)} chunks from {source_file} to the collection.")
    except Exception as e:
        print(f"Error adding documents to collection: {str(e)}")

def get_relevant_context(collection, model, query_text, top_k=3):
    """Retrieve relevant context for the query."""
    try:
        query_embedding = model.encode([query_text])[0].tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        context = "\n\n".join(results['documents'][0])
        return context
    except Exception as e:
        print(f"Error getting context: {str(e)}")
        return ""

def generate_rag_response(query, context):
    """Generate a response using GPT-4 with the retrieved context."""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Use the provided context to answer the user's question. If the context doesn't contain relevant information, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def process_files():
    try:
        if not os.path.exists(TEXT_FILES_DIR):
            raise ValueError(f"Text files directory not found at {TEXT_FILES_DIR}")

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        collection = get_or_create_collection(chroma_client, "text_collection")

        # Process text files
        txt_files = [f for f in os.listdir(TEXT_FILES_DIR) if f.endswith('.txt')]
        if not txt_files:
            print(f"No .txt files found in {TEXT_FILES_DIR}")
            return

        print(f"Found {len(txt_files)} .txt files to process")
        
        for filename in txt_files:
            file_path = os.path.join(TEXT_FILES_DIR, filename)
            print(f"\nProcessing file: {filename}")
            text_chunks = read_and_chunk_text(file_path)
            if text_chunks:
                add_documents_to_collection(collection, model, text_chunks, file_path)
            else:
                print(f"No content extracted from {filename}")

    except Exception as e:
        print(f"Error processing files: {str(e)}")
        raise

def interactive_session():
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        collection = get_or_create_collection(chroma_client, "text_collection")
        print("\nRAG System Ready. Enter your questions (type 'exit' to quit):")
        
        while True:
            query = input("\nYour question: ").strip()
            
            if query.lower() in ['exit', 'quit']:
                break
                
            if not query:
                continue
                
            context = get_relevant_context(collection, model, query)
            response = generate_rag_response(query, context)
            print("\nResponse:", textwrap.fill(response, width=100))
            
    except Exception as e:
        print(f"Error in interactive session: {str(e)}")

def main():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        process_files()
        interactive_session()

    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()