from document_parser import DocumentParser
from text_chunker import TextChunker
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from semantic_extractor import SemanticComponentExtractor
from config import CHUNK_SIZE, CHUNK_OVERLAP
import json

class SemanticDocumentIntelligence:
    def __init__(self):
        self.parser = DocumentParser()
        self.chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.extractor = SemanticComponentExtractor(self.embedder)  # INTEGRATE HERE
        self.ingested_docs = []
    
    def ingest_document(self, file_path):
        """Parse → Chunk → Embed → Store"""
        print(f"\nProcessing: {file_path}")
        
        text, metadata = self.parser.parse_document(file_path)
        if text is None:
            return False
        
        chunks = self.chunker.chunk_text(text, document_name=metadata['filename'])
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedder.encode_batch(chunk_texts)
        
        self.vector_store.add_documents(chunks, embeddings)
        self.ingested_docs.append({
            'filename': metadata['filename'],
            'chunks': len(chunks),
            'metadata': metadata
        })
        return True
    
    def semantic_search(self, query, top_k=5):
        """Search documents"""
        query_embedding = self.embedder.encode_single(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        formatted = []
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            formatted.append({
                'rank': i + 1,
                'score': round(1 - dist, 4),
                'document': meta['document'],
                'chunk_id': meta['chunk_id'],
                'content': doc[:200] + "..." if len(doc) > 200 else doc
            })
        return formatted
    
    def analyze_document(self, file_path):
        """NEW: Use semantic extractor"""
        from document_parser import DocumentParser
        parser = DocumentParser()
        text, _ = parser.parse_document(file_path)
        if text:
            return self.extractor.generate_semantic_report(text, file_path)
        return None
    
    def get_system_info(self):
        """System status"""
        return {
            'ingested_documents': len(self.ingested_docs),
            'embedding_model': self.embedder.model_name,
            'vector_db': 'Chroma (Local)',
            'device': str(self.embedder.model.device)
        }

# Initialization function
def initialize_system():
    system = SemanticDocumentIntelligence()
    print("✓ System initialized")
    print(json.dumps(system.get_system_info(), indent=2))
    return system