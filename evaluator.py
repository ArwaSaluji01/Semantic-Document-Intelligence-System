import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

class SystemEvaluator:
    """Evaluate semantic extraction and search quality"""
    
    def __init__(self, system):
        self.system = system
        self.results = {}
    
    def evaluate_extraction_quality(self, file_path, expected_components):
        """Test semantic extraction accuracy
        
        Args:
            file_path: Document to analyze
            expected_components: Dict with expected keys (sections, entities, metrics, etc.)
        """
        print("\nEXTRACTION QUALITY EVALUATION")
        print("="*60)
        
        result = self.system.analyze_document(file_path)
        
        if not result:
            print("Extraction failed")
            return {"status": "failed"}
        
        scores = {}
        
        # Check sections extraction
        extracted_sections = len(result.get('sections', {}))
        expected_sections = expected_components.get('sections', 0)
        section_score = min(1.0, extracted_sections / max(expected_sections, 1))
        scores['section_extraction'] = section_score
        print(f"Sections: {extracted_sections}/{expected_sections} ({section_score*100:.1f}%)")
        
        # Check entities extraction
        total_entities = sum(len(v) for v in result.get('entities', {}).values())
        expected_entities = expected_components.get('entities', 0)
        entity_score = min(1.0, total_entities / max(expected_entities, 1))
        scores['entity_extraction'] = entity_score
        print(f"Entities: {total_entities}/{expected_entities} ({entity_score*100:.1f}%)")
        
        # Check metrics extraction
        total_metrics = sum(len(v) for v in result.get('metrics', {}).values())
        expected_metrics = expected_components.get('metrics', 0)
        metric_score = min(1.0, total_metrics / max(expected_metrics, 1))
        scores['metric_extraction'] = metric_score
        print(f"Metrics: {total_metrics}/{expected_metrics} ({metric_score*100:.1f}%)")
        
        # Use cases
        extracted_usecases = len(result.get('use_cases', []))
        expected_usecases = expected_components.get('use_cases', 0)
        usecase_score = min(1.0, extracted_usecases / max(expected_usecases, 1))
        scores['usecase_extraction'] = usecase_score
        print(f"Use Cases: {extracted_usecases}/{expected_usecases} ({usecase_score*100:.1f}%)")
        
        # Problem-Solution pairs
        ps_pairs = len(result.get('problems_solutions', []))
        expected_ps = expected_components.get('problems_solutions', 0)
        ps_score = min(1.0, ps_pairs / max(expected_ps, 1))
        scores['ps_extraction'] = ps_score
        print(f"P-S Pairs: {ps_pairs}/{expected_ps} ({ps_score*100:.1f}%)")
        
        # Overall extraction quality
        avg_score = np.mean(list(scores.values()))
        print(f"\nOverall Extraction Quality: {avg_score*100:.1f}%")
        
        self.results['extraction_quality'] = scores
        return scores
    
    def evaluate_semantic_search(self, queries_with_expected):
        """Test search quality with expected results
        
        Args:
            queries_with_expected: List of dicts with 'query' and 'expected_docs'
        """
        print("\n🔍 SEMANTIC SEARCH EVALUATION")
        print("="*60)
        
        precision_scores = []
        recall_scores = []
        
        for item in queries_with_expected:
            query = item['query']
            expected_docs = set(item['expected_docs'])
            
            # Get results
            results = self.system.semantic_search(query, top_k=5)
            retrieved_docs = set([r['document'] for r in results])
            
            # Calculate metrics
            if retrieved_docs:
                precision = len(expected_docs & retrieved_docs) / len(retrieved_docs)
            else:
                precision = 0
            
            recall = len(expected_docs & retrieved_docs) / len(expected_docs) if expected_docs else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            print(f"\nQuery: '{query}'")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall: {recall:.3f}")
        
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        print(f"\nAverage Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"F1-Score: {f1_score:.3f}")
        
        self.results['search_quality'] = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1_score
        }
        
        return self.results['search_quality']
    
    def evaluate_embedding_quality(self):
        """Test embedding model quality"""
        print("\nEMBEDDING QUALITY EVALUATION")
        print("="*60)
        
        # Test semantic similarity
        similar_pairs = [
            ("machine learning", "ML algorithms"),
            ("neural networks", "deep learning"),
            ("vector database", "semantic search")
        ]
        
        similarity_scores = []
        for text1, text2 in similar_pairs:
            emb1 = self.system.embedder.encode_single(text1)
            emb2 = self.system.embedder.encode_single(text2)
            
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            similarity_scores.append(similarity)
            print(f"'{text1}' vs '{text2}': {similarity:.3f}")
        
        avg_similarity = np.mean(similarity_scores)
        print(f"\nAverage Semantic Similarity: {avg_similarity:.3f}")
        
        self.results['embedding_quality'] = {
            'avg_similarity': avg_similarity,
            'embedding_dim': self.system.embedder.embedding_dim
        }
        
        return self.results['embedding_quality']
    
    def generate_full_report(self):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print("SYSTEM EVALUATION REPORT")
        print("="*60)
        
        report = {
            'extraction_quality': self.results.get('extraction_quality', {}),
            'search_quality': self.results.get('search_quality', {}),
            'embedding_quality': self.results.get('embedding_quality', {}),
        }
        
        # Calculate overall score (0-100)
        scores = []
        if report['extraction_quality']:
            scores.append(np.mean(list(report['extraction_quality'].values())))
        if report['search_quality']:
            scores.append(report['search_quality'].get('f1_score', 0))
        if report['embedding_quality']:
            scores.append(min(report['embedding_quality'].get('avg_similarity', 0), 1.0))
        
        overall_score = np.mean(scores) * 100 if scores else 0
        
        print(f"\nOVERALL SYSTEM PERFORMANCE: {overall_score:.1f}/100")
        print(json.dumps(report, indent=2, default=str))
        
        return report, overall_score