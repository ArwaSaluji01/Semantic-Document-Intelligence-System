import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Text Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Vector Store
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "documents"

# Semantic Extraction
SUPPORTED_FORMATS = ['.pdf', '.docx', '.txt']