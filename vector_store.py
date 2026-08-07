import chromadb
from config import PERSIST_DIR, COLLECTION_NAME

class VectorStore:
    def __init__(self, persist_dir=PERSIST_DIR):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ Vector store initialized at: {persist_dir}")
    
    def add_documents(self, chunks, embeddings):
        """Add chunks and embeddings to database"""
        ids = []
        metadatas = []
        documents = []
        
        for i, chunk in enumerate(chunks):
            ids.append(f"{chunk['document']}_chunk_{chunk['chunk_id']}")
            documents.append(chunk['content'])
            metadatas.append({
                'document': chunk['document'],
                'chunk_id': str(chunk['chunk_id']),
                'chunk_size': str(chunk['chunk_size'])
            })
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✓ Added {len(chunks)} chunks to vector store")
    
    def search(self, query_embedding, top_k=5):
        """Semantic search in vector store"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        return results
    
    def get_collection_info(self):
        """Get database statistics"""
        count = self.collection.count()
        return {'total_documents': count}

# Initialize vector store
vector_store = VectorStore(persist_dir="./chroma_db")
print(f"Collection info: {vector_store.get_collection_info()}")