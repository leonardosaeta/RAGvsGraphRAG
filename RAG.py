import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

DATA_DIR = "./chroma_db"
os.makedirs(DATA_DIR, exist_ok=True)

def get_or_create_collection(client, collection_name):
    """Check if a collection exists, and create it if not."""
    existing_collections = client.list_collections()
    
    for collection in existing_collections:
        if collection['name'] == collection_name:
            print(f"Collection '{collection_name}' already exists.")
            return client.get_collection(collection_name)
    
    print(f"Creating collection '{collection_name}'.")
    return client.create_collection(collection_name)

def add_documents_to_collection(collection, model, documents):
    """Generate embeddings for documents and add them to the collection."""
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(texts)

    collection.add(
        embeddings=embeddings.tolist(),  
        documents=texts,                 
        ids=[doc["id"] for doc in documents],  
        metadatas=[doc["metadata"] for doc in documents]  
    )
    print(f"Added {len(documents)} documents to the collection.")

def query_collection(collection, model, query_text, top_k=2):
    """Encode a query and retrieve the most similar documents from the collection."""
    query_embedding = model.encode([query_text])[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    documents = results['documents'][0]   
    metadatas = results['metadatas'][0]  
    distances = results['distances'][0]   
    ids = results['ids'][0]              
    
    for doc, meta, distance, doc_id in zip(documents, metadatas, distances, ids):
        print(f"Document ID: {doc_id}")
        print(f"Distance: {distance}")
        print(f"Metadata: {meta}")
        print(f"Content: {doc}")
        print("\n")

def main():
    try:
        # Initialize persistent client
        client = chromadb.PersistentClient(path=DATA_DIR)
        collection_name = "my_collection"
        
        existing_collections = client.list_collections()
        collection_exists = any(c.name == collection_name for c in existing_collections)
        
        collection = client.get_or_create_collection(collection_name)
        
        print("Loading model from Hugging Face hub...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        if not collection_exists:
            print("Creating new collection and adding documents...")
            documents = [
                {"id": "doc1", "text": "This is the first document.", "metadata": {"source": "web"}},
                {"id": "doc2", "text": "Second document content here.", "metadata": {"source": "books"}},
                {"id": "doc3", "text": "Another piece of information in document three.", "metadata": {"source": "articles"}},
            ]
            add_documents_to_collection(collection, model, documents)
        else:
            print("Using existing collection from disk...")

        query_text = "Find information about the first document"
        query_collection(collection, model, query_text, top_k=2)

    except Exception as e:
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
