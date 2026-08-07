from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextChunker:
    """Split documents into manageable chunks"""
    
    def __init__(self, chunk_size=500, chunk_overlap=100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text, document_name=""):
        """Split text into chunks with metadata"""
        chunks = self.text_splitter.split_text(text)
        
        # Add metadata to each chunk
        chunks_with_metadata = []
        for idx, chunk in enumerate(chunks):
            chunks_with_metadata.append({
                'content': chunk,
                'chunk_id': idx,
                'document': document_name,
                'chunk_size': len(chunk)
            })
        
        return chunks_with_metadata

# Initialize chunker
chunker = TextChunker(chunk_size=500, chunk_overlap=100)
print("✓ Text Chunker initialized")
print(f"  - Chunk size: 500 tokens")
print(f"  - Overlap: 100 tokens")