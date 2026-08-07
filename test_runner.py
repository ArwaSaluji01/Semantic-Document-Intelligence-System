from main import initialize_system
from test_data import create_test_files
from evaluator import SystemEvaluator

def run_basic_tests():
    """Test ingestion & search"""
    print("\n" + "="*60)
    print("TEST 1: DOCUMENT INGESTION & SEARCH")
    print("="*60)
    
    system = initialize_system()
    test_files = create_test_files()
    
    # Ingest
    for file in test_files:
        system.ingest_document(file)
    
    # Search
    queries = [
        "What is machine learning?",
        "How does semantic search work?"
    ]
    
    for query in queries:
        results = system.semantic_search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        for r in results:
            print(f"  Rank {r['rank']} (Score: {r['score']}) - {r['content'][:100]}")

def run_semantic_extraction_tests():
    """Test semantic component extraction"""
    print("\n" + "="*60)
    print("TEST 2: SEMANTIC EXTRACTION")
    print("="*60)
    
    system = initialize_system()
    create_test_files()
    
    # Analyze rich document
    result = system.analyze_document("data/test_docs/rich_semantic_doc.txt")
    
    if result:
        print("\n✓ Semantic extraction complete")
        print(f"  Sections: {len(result['sections'])}")
        print(f"  Entities: {len(result['entities'])}")
        print(f"  Use Cases: {len(result['use_cases'])}")

from evaluator import SystemEvaluator

def run_evaluation_tests():
    """Run comprehensive evaluation"""
    print("\n" + "="*60)
    print("TEST 3: COMPREHENSIVE SYSTEM EVALUATION")
    print("="*60)
    
    system = initialize_system()
    create_test_files()
    
    # Ingest all test files
    files = [
        "data/test_docs/neural_networks.txt",
        "data/test_docs/knowledge_management.txt",
        "data/test_docs/research_paper.txt"
    ]
    
    for f in files:
        system.ingest_document(f)
    
    evaluator = SystemEvaluator(system)
    
    # 1. Extraction quality
    expected_components_1 = {
        'sections': 7,
        'entities': 15,
        'metrics': 12,
        'use_cases': 5,
        'problems_solutions': 4
    }
    evaluator.evaluate_extraction_quality(files[0], expected_components_1)
    
    # 2. Search quality
    queries_with_expected = [
        {
            'query': 'How do transformers work?',
            'expected_docs': {'neural_networks.txt', 'research_paper.txt'}
        },
        {
            'query': 'What is semantic search?',
            'expected_docs': {'knowledge_management.txt', 'research_paper.txt'}
        },
        {
            'query': 'Machine learning performance metrics',
            'expected_docs': {'neural_networks.txt', 'research_paper.txt'}
        }
    ]
    evaluator.evaluate_semantic_search(queries_with_expected)
    
    # 3. Embedding quality
    evaluator.evaluate_embedding_quality()
    
    # 4. Full report
    report, score = evaluator.generate_full_report()
    
    return report, score

if __name__ == "__main__":
    run_basic_tests()
    run_semantic_extraction_tests()
    run_evaluation_tests()
    print("\n✓ All tests completed successfully!")