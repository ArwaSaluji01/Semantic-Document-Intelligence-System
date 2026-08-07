from collections import defaultdict
import re
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")

class SemanticComponentExtractor:
    """Extract all semantic components from documents"""
    def __init__(self, embedder):
        self.embedder = embedder

    # Define semantic patterns
        self.patterns = {
            'entities': {
                'organization': r'(?:Organization|Company|Firm|Enterprise|Corporation|Team|Department)s?:?\s*([A-Z][^,\n]+)',
                'person': r'(?:Manager|Officer|Engineer|Analyst|Researcher|Director|CEO|CTO)s?:?\s*([A-Z][^\n]+)',
                'technology': r'(?:Model|Database|Tool|Framework|Algorithm|System|Platform|Service)s?:?\s*([A-Za-z0-9\s\-]+?)(?=\n|,|$)',
            },
            'metrics': {
                'performance': r'(?:Accuracy|Precision|Recall|Latency|Throughput|F1|mAP|BLEU|ROUGE|WER)[\s:]*([0-9.%]+)',
                'business': r'(?:Cost (?:Savings|Reduction)|ROI|Revenue|Time (?:Savings|Reduction)|Improvement|Growth)[\s:]*([^\n]+?)(?=\n|$)',
            },
        }
    
    def extract_sections(self, text):
        """Extract document sections"""
        sections = re.split(r'\n(?=\d+\.?\s+[A-Z]|\s{0,2}[A-Z][A-Z\s]+:)', text)
        extracted = {}
        for i, section in enumerate(sections[1:], 1):
            title = section.split('\n')[0][:50]
            extracted[f'Section_{i}'] = {'title': title.strip(), 'content': section.strip()}
        return extracted
    
    def extract_entities(self, text):
        entities = defaultdict(list)
        doc = nlp(text)
        
        # spaCy NER
        for ent in doc.ents:
            if ent.label_ == "ORG":
                entities['organization'].append(ent.text)
            elif ent.label_ == "PERSON":
                entities['person'].append(ent.text)
        
        # Tech keywords (regex fallback)
        tech_keywords = r'\b(?:PyTorch|Transformer|YOLO|BERT|ResNet|Elasticsearch|Pinecone|AutoML)\b'
        entities['technology'].extend(re.findall(tech_keywords, text))
        
        return {k: list(set(v)) for k, v in entities.items()}
    
    def extract_metrics(self, text):
        metrics = defaultdict(list)
        
        # Pattern: metric_name: number + unit
        pattern = r'(?:Accuracy|Precision|Recall|F1|mAP|BLEU|ROUGE|WER|Latency|Throughput|ROI|Savings|Improvement|Growth)[\s:]+([0-9.%]+(?:\s*(?:ms|sec|hours?|%|x|$))?)'
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = match.group(1).strip()
            metrics['performance'].append(value)
        
        return dict(metrics)
    
    def extract_problems_solutions(self, text):
        ps_pairs = []
        blocks = re.findall(
            r'Challenge.*?:\s*(.+?)\n.*?(?:Solution|Fix).*?:\s*(.+?)(?=\n\n|Challenge|$)',
            text, re.IGNORECASE | re.DOTALL
        )
        for problem, solution in blocks:
            ps_pairs.append({'problem': problem.strip()[:100], 'solution': solution.strip()[:100]})
        return ps_pairs
    
    def extract_use_cases(self, text):
        use_cases = []
        
        # Match "Use Case X: Title" + content until next section
        pattern = r'Use Case \d+:\s*([^\n]+)\n([\s\S]*?)(?=Use Case|^\d+\.|$)'
        
        for match in re.finditer(pattern, text, re.MULTILINE):
            title = match.group(1).strip()
            content = match.group(2)
            
            use_cases.append({
                'title': title,
                'query': re.search(r'(?:User )?[Qq]uery:?\s*"([^"]+)"', content),
                'result': re.search(r'[Rr]esult:?\s*(.+?)(?=\n|$)', content),
                'impact': re.search(r'(?:Impact|Time|Accuracy|Precision):\s*(.+?)(?=\n|$)', content),
            })
        
        return [{k: (v.group(1).strip() if v else 'N/A') for k, v in uc.items()} for uc in use_cases]
    
    def semantic_fingerprint(self, text):
        """Create vector fingerprint of semantic content"""
        # Extract key phrases
        sentences = text.split('.')
        key_sentences = [s.strip() for s in sentences if len(s.split()) > 5][:10]
        
        # Embed and average
        embeddings = self.embedder.encode_batch(key_sentences)
        fingerprint = np.mean(embeddings, axis=0)
        
        return {
            'fingerprint': fingerprint,
            'dimension': len(fingerprint),
            'key_phrases': key_sentences
        }
    
    def analyze_semantic_density(self, text):
        """Measure semantic richness"""
        sections = len(re.findall(r'\d+\. SEMANTIC LAYER:', text))
        entities = len(re.findall(r':\s*[A-Z][^:]+?(?=\n|$)', text))
        metrics = len(re.findall(r'(?:Accuracy|Precision|Recall|Latency|ROI|Savings):', text))
        problems = len(re.findall(r'Challenge \d+:', text))
        
        return {
            'sections': sections,
            'entities': entities,
            'metrics': metrics,
            'problems_solutions': problems,
            'semantic_density_score': (sections + entities + metrics + problems) / (len(text) / 1000)
        }
    
    def generate_semantic_report(self, text, doc_name="document"):
        """Generate complete semantic analysis"""
        print(f"\n{'='*70}")
        print(f"SEMANTIC ANALYSIS: {doc_name}")
        print(f"{'='*70}")
        
        # 1. Sections
        print("\nDOCUMENT STRUCTURE:")
        sections = self.extract_sections(text)
        for sec_id, sec_data in sections.items():
            print(f"  {sec_id}: {sec_data['title']}")
        
        # 2. Entities
        print("\nEXTRACTED ENTITIES:")
        entities = self.extract_entities(text)
        for ent_type, values in entities.items():
            print(f"  {ent_type.upper()}: {', '.join(values[:3])}")
        
        # 3. Metrics
        print("\nEXTRACTED METRICS:")
        metrics = self.extract_metrics(text)
        for metric_type, values in metrics.items():
            print(f"  {metric_type.upper()}: {', '.join(values[:2])}")
        
        # 4. Problems & Solutions
        print("\nPROBLEMS & SOLUTIONS:")
        ps = self.extract_problems_solutions(text)
        for i, pair in enumerate(ps[:2], 1):
            print(f"  Problem {i}: {pair['problem']}")
            print(f"  Solution {i}: {pair['solution']}")
        
        # 5. Use Cases
        print("\nUSE CASES:")
        use_cases = self.extract_use_cases(text)
        for uc in use_cases[:2]:
            print(f"  - {uc['title']}")
            print(f"    Input: {uc['input']}")
            print(f"    Output: {uc['output']}")
        
        # 6. Semantic Fingerprint
        print("\nSEMANTIC FINGERPRINT:")
        fp = self.semantic_fingerprint(text)
        print(f"  Embedding Dimension: {fp['dimension']}")
        print(f"  Key Phrases: {', '.join(fp['key_phrases'][:3])}")
        
        # 7. Semantic Density
        print("\nSEMANTIC DENSITY METRICS:")
        density = self.analyze_semantic_density(text)
        for key, value in density.items():
            print(f"  {key.upper()}: {value}")
        
        print(f"\n{'='*70}\n")
        
        return {
            'sections': sections,
            'entities': entities,
            'metrics': metrics,
            'problems_solutions': ps,
            'use_cases': use_cases,
            'fingerprint': fp,
            'density': density
        } 