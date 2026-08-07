from sentence_transformers import SentenceTransformer
from config import DEVICE, EMBEDDING_MODEL

class EmbeddingGenerator:
    def __init__(self, model_name=EMBEDDING_MODEL):
        print(f"Loading: {model_name}")
        self.model = SentenceTransformer(model_name, device=DEVICE)
        self.model_name = model_name
        self.embedding_dim = self.model.get_embedding_dimension()
        print(f"✓ Model loaded | Embedding dimension: {self.embedding_dim}")
    
    def encode_batch(self, texts, batch_size=32):
        """Encode multiple texts efficiently"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_single(self, text):
        """Encode single text"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def get_stats(self):
        """Print model statistics"""
        return {
            'model': self.model_name,
            'dimension': self.embedding_dim,
            'device': str(DEVICE)
        }