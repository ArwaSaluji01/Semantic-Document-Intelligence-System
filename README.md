# Semantic Document Intelligence System

An end-to-end pipeline for intelligent document processing, semantic extraction, and contextual search using embeddings and vector databases.

## Overview

Process unstructured documents → Extract semantic components (entities, metrics, problems/solutions) → Index embeddings → Perform intelligent semantic search with 86% F1-score.

**Current Performance: 67/100** (Extraction: 58.3% | Search F1: 0.86 | Embeddings: 0.568 similarity)

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```python
from main import initialize_system

# Initialize
system = initialize_system()

# Ingest documents
system.ingest_documents(['data/test_docs/neural_networks.txt'])

# Semantic search
results = system.semantic_search('How do transformers work?', top_k=3)
for r in results:
    print(f"Score: {r['score']:.3f} | {r['document'][:100]}")

# Extract semantic components
analysis = system.analyze_document('data/test_docs/neural_networks.txt')
print(f"Entities: {analysis['entities']}")
print(f"Metrics: {analysis['metrics']}")
print(f"Problems: {analysis['problems_solutions']}")
```

## Evaluation Results

| Metric | Score |
|--------|-------|
| **Section Extraction** | 100.0% |
| **Entity Extraction** | 100.0% |
| **Metric Extraction** | 41.7% |
| **Use Case Extraction** | 0.0% |
| **Problem-Solution Pairs** | 50.0% |
| **Search Precision** | 88.9% |
| **Search Recall** | 83.3% |
| **Search F1-Score** | 0.86 |
| **Avg Embedding Similarity** | 0.568 |
| **Overall System Score** | 67.0/100 |

## Configuration

Edit `config.py`:
```python
CHUNK_SIZE = 500           # Tokens per chunk
CHUNK_OVERLAP = 100        # Overlap tokens
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB_PATH = "./chroma_db"
```

## Run Tests

```bash
python test_runner.py
```

Tests ingestion, search quality, extraction accuracy, and embedding performance across 3 domain documents (neural networks, knowledge management, research papers).

## Key Features

✅ **Multi-format parsing** — TXT, PDF, DOCX  
✅ **Smart chunking** — Token-aware overlap for semantic coherence  
✅ **Semantic embeddings** — 384-D vectors, 0.568 avg similarity  
✅ **Vector search** — Precision 88.9%, Recall 83.3%  
✅ **Entity extraction** — ORGANIZATION, PERSON, TECHNOLOGY  
✅ **Metrics capture** — Performance metrics (accuracy, F1, mAP)  
✅ **Problem-solution pairs** — Structured challenge extraction  
✅ **Semantic fingerprints** — Key phrases + embedding vectors  

## Future Improvements

- [ ] Use case extraction (currently 0%)
- [ ] Entity type disambiguation (PERSON/ORGANIZATION confusion)
- [ ] Metric unit normalization (% vs decimal)
- [ ] Multi-hop semantic reasoning
- [ ] Fine-tuned domain embeddings
- [ ] LLM-based answer generation
