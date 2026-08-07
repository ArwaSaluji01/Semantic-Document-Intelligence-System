import os

TEST_DOC_1 = """
NEURAL NETWORKS AND DEEP LEARNING ARCHITECTURE

Executive Summary:
Deep learning revolutionizes AI by using multi-layer neural networks. 
Organizations using DL see 40% efficiency improvements.

1. TECHNICAL FOUNDATIONS

Architecture Components:
- Input Layer: Receives raw data (images, text, numbers)
- Hidden Layers: Process information through learned weights
- Output Layer: Produces predictions or classifications

Activation Functions:
- ReLU: Best for hidden layers, prevents vanishing gradient
- Sigmoid: Used for binary classification (0-1 output)
- Softmax: Multi-class classification (probability distribution)

2. TRAINING MECHANISMS

Backpropagation Algorithm:
1. Forward pass: Data flows through network
2. Calculate loss: Measure prediction error
3. Backward pass: Compute gradients
4. Update weights: Optimizer adjusts parameters

Common Optimizers:
- SGD: Basic gradient descent, slow convergence
- Adam: Adaptive learning, industry standard
- RMSprop: Good for RNN, prevents overshooting

3. PERFORMANCE METRICS

Accuracy: (TP + TN) / (TP + TN + FP + FN)
- Classification accuracy reached: 94.7%
- Validation accuracy: 91.2%

Precision & Recall:
- Precision: How many predicted positives are actually positive
- Recall: How many actual positives are found
- F1-Score: Harmonic mean of precision and recall (0.893)

Loss Metrics:
- Training loss: 0.234 (decreasing trend ✓)
- Validation loss: 0.312 (slight overfitting)
- Final convergence: Epoch 150

4. CHALLENGES & SOLUTIONS

Challenge: Vanishing Gradient Problem
- Issue: Gradients become too small in deep networks
- Impact: Lower layers stop learning
- Solution: Use ReLU activation, batch normalization

Challenge: Overfitting
- Symptom: High train accuracy, low validation accuracy
- Cause: Model memorizes training data
- Solutions: Dropout (50%), L2 regularization, early stopping

5. REAL-WORLD APPLICATIONS

Computer Vision:
- Image Classification: ResNet, VGG16, EfficientNet
- Object Detection: YOLO, Faster R-CNN (mAP: 0.85)
- Segmentation: U-Net for medical imaging (Dice: 0.92)

Natural Language Processing:
- Machine Translation: Transformer models
- Sentiment Analysis: BERT-based (F1: 0.91)
- Named Entity Recognition: BiLSTM-CRF (precision: 0.89)

Time Series:
- LSTM networks for forecasting
- Prediction accuracy: 87.3% on stock data
- RMSE: 2.34 on normalized scale

6. DEPLOYMENT CONSIDERATIONS

Model Compression:
- Quantization: Reduce from 32-bit to 8-bit (4x smaller)
- Pruning: Remove 30% of parameters with <1% accuracy loss
- Distillation: Transfer learning to smaller model (95% performance)

Inference Latency:
- GPU inference: 12ms per batch (1000 images/sec)
- CPU inference: 85ms per batch (fallback option)
- Edge device (mobile): 200ms acceptable for real-time

Monitoring & Maintenance:
- Track model drift: Accuracy drops below 88% → retrain
- Data quality monitoring: Flag anomalies
- A/B testing: New model vs production (confidence threshold: 95%)

7. FUTURE TRENDS

Federated Learning:
- Train across distributed devices
- Privacy-preserving: No raw data centralization
- Use case: Healthcare, finance

Quantum Machine Learning:
- Quantum circuits for optimization
- Potential: 100x speedup for certain problems
- Status: Research phase

AutoML & Neural Architecture Search:
- Automatically design optimal network architecture
- Reduce manual tuning time by 80%
- Tools: AutoKeras, Google AutoML
"""

TEST_DOC_2 = """
ENTERPRISE KNOWLEDGE MANAGEMENT SYSTEMS

Business Context:
Global Fortune 500 company managing 50M+ documents. Challenge: Enable 
employees to find relevant information in seconds, not hours.

1. SYSTEM ARCHITECTURE

Components:
- Document Ingestion: 10K docs/day across all departments
- Indexing: Real-time with <2 second latency
- Search Engine: Multi-modal (text, images, metadata)
- Analytics: Track usage patterns and user behavior

Technology Stack:
- Storage: Elasticsearch for full-text search
- Vector DB: Pinecone for semantic search
- Cache: Redis (99.9% hit rate)
- API: GraphQL for flexible querying

2. KNOWLEDGE DOMAINS

HR Documentation:
- Employee policies: 500+ documents
- Benefits information: Medical, dental, retirement
- Training materials: Onboarding, compliance courses
- Avg retrieval time improvement: 75% reduction

Finance & Compliance:
- Regulatory documents: GDPR, SOC2, HIPAA
- Tax filings: Historical records (15 years)
- Audit trails: Complete change history
- Compliance score: 99.7%

Product Documentation:
- Technical specifications: 3000+ pages
- API documentation: Auto-generated from code
- User guides: 25 languages
- Search accuracy: 96.2%

3. KEY PERFORMANCE INDICATORS

User Engagement:
- Daily active users: 12,500 (85% of workforce)
- Avg session duration: 4.2 minutes
- Search queries per user: 8.3 daily
- User satisfaction: 4.6/5.0 stars

System Performance:
- Query response time: 234ms (p95)
- Search result quality: 92.1% relevance
- Index update latency: 500ms
- System uptime: 99.98%

Business Impact:
- Time savings: 15 hours/employee/year
- Cost reduction: $2.1M annually
- Employee productivity: +18% improvement
- Customer satisfaction: +12% (faster support)

4. CHALLENGES ENCOUNTERED

Challenge 1: Information Silos
- Multiple systems storing duplicate data
- Inconsistent metadata standards
- Solution: Unified metadata schema, deduplication engine

Challenge 2: Semantic Ambiguity
- "Interest" = Investment returns OR attention/focus
- "Bank" = Financial institution OR river embankment
- Solution: Named entity recognition + context window expansion

Challenge 3: Scalability at Peak Hours
- Search queries spike 3x during business hours
- Latency degradation observed
- Solution: Load balancing, intelligent caching, query optimization

Challenge 4: Data Quality Issues
- Outdated documents still appearing in results
- Corrupted metadata in legacy systems
- Solution: Automated deprecation, quality scoring system

5. SEMANTIC SEARCH IMPLEMENTATION

Vector Embeddings:
- Model: multi-qa-MiniLM-L6-cos-v1 (384-D)
- Training data: 100K domain-specific documents
- Custom fine-tuning: +8% accuracy improvement

Hybrid Search Strategy:
- Combine BM25 (keyword) + Vector (semantic)
- Weight: 40% keyword + 60% semantic
- Fallback: Pure keyword if vectors insufficient

Result Re-ranking:
- Cross-encoder model for precision
- Rank top-50 semantic results
- Final relevance score: Correlation with user clicks (r=0.87)

6. USE CASES & SUCCESS STORIES

Use Case 1: Legal Compliance Query
- User query: "Which documents mention GDPR Article 32?"
- Traditional search: 2000+ false positives (keywords scattered)
- Semantic search: 47 highly relevant documents (precision: 0.98)
- Time saved: 3 hours/query

Use Case 2: Customer Support Automation
- Support agent asks: "How do I reset customer API key?"
- System retrieves: FAQ, API docs, troubleshooting guide
- Accuracy: 94.3% useful results
- First-contact resolution rate: 73%

Use Case 3: Cross-departmental Insights
- Manager query: "What are our cloud cost optimization initiatives?"
- Aggregates: Finance reports, ops docs, vendor proposals
- Enables data-driven decision making
- ROI improvements: +23%

7. FUTURE ROADMAP

Q1 2024: Generative AI Integration
- Add LLM-based answer generation
- Summarization of multi-document results
- Estimated impact: 40% more user queries

Q2 2024: Multi-modal Search
- Image search capability
- Document diagrams/charts indexing
- Expected engagement: +25%

Q3 2024: Predictive Search
- Anticipate user needs before query
- ML model trained on click patterns
- Projected efficiency: 30% fewer searches needed
"""

TEST_DOC_3 = """
RESEARCH PAPER: ADVANCES IN TRANSFORMER ARCHITECTURES

Abstract:
This paper reviews recent advances in transformer-based models and their 
impact on AI research. We analyze 150+ papers published in 2023-2024, 
identifying key trends and breakthrough applications.

1. INTRODUCTION & MOTIVATION

Background:
- Transformers introduced by Vaswani et al. (2017)
- Revolutionized NLP, vision, and multimodal AI
- Adoption: 90% of SOTA models now use transformers

Research Questions:
- How have transformer architectures evolved?
- What are the scaling laws and efficiency improvements?
- Which applications show highest impact?
- What are remaining limitations?

2. ARCHITECTURE INNOVATIONS

Vision Transformers (ViT):
- Pure transformer for image classification
- Performance: 86.7% ImageNet accuracy
- Advantages: Better transfer learning than CNNs
- Trade-off: Requires more pre-training data

Efficient Transformers:
- Linear attention mechanisms (instead of quadratic)
- Sparse attention patterns
- Inference speedup: 2.5-4x
- Accuracy retention: 98.1% (marginal loss)

Multimodal Transformers:
- CLIP: Vision + Language alignment
- Performance: CLIP benchmark score 76.2%
- Applications: Zero-shot image classification, retrieval
- Impact: Reduced labeled data requirements by 60%

3. SCALING LAWS & TRENDS

Model Size Progression:
- 2020: GPT-3 (175B parameters)
- 2022: PaLM (540B parameters)
- 2023: Gemini (1.4T parameters)
- Trend: +400% growth in 3 years

Performance vs Scale:
- Compute requirements: Double per year (Moore's Law)
- Accuracy improvement: Log-linear with scale
- Emergent capabilities appear at ~60B parameters
- Cost per token: Decreasing 20% annually

Data Efficiency:
- Pre-training data: 2T tokens (GPT-4)
- Fine-tuning reduction: Achievable with 1% of labeled data
- Few-shot learning: Viable with 5-10 examples

4. APPLICATION DOMAINS

Natural Language Processing:
- Machine translation: BLEU score 32.5 (SOTA)
- Summarization: ROUGE-L 47.2
- Question answering: F1 score 94.1
- Sentiment analysis: Accuracy 97.3%

Computer Vision:
- Image classification: Top-1 accuracy 90.2%
- Object detection: mAP 58.9
- Instance segmentation: Mask mAP 52.4
- 3D scene understanding: New frontier

Multimodal Learning:
- Image captioning: BLEU score 41.3
- Visual question answering: Accuracy 78.6%
- Video understanding: Temporal modeling advances
- Cross-modal retrieval: mAP 42.1

Speech & Audio:
- Speech recognition: WER 5.2% (conversational)
- Speaker recognition: Accuracy 98.7%
- Music generation: Evaluated by human raters (8.2/10)
- Audio captioning: Novel task, CIDEr score 68.3

5. EFFICIENCY & DEPLOYMENT

Model Compression Techniques:
- Quantization: 4-bit (INT4) with <2% accuracy loss
- Distillation: Teacher (700M) → Student (70M) with 96% performance
- Pruning: Remove 40% weights, maintain 99% accuracy
- Rank reduction: Low-rank factorization for faster inference

Inference Optimization:
- GPU inference: 50ms latency per token (A100)
- CPU inference: 400ms (acceptable for batch processing)
- Edge devices: 2-5 second latency (mobile optimization needed)
- Batch processing: 10x throughput improvement vs streaming

Cost Analysis:
- Training cost: $10M for 1.4T parameter model
- Fine-tuning: $10K-$100K per domain
- Inference cost: $0.001 per 1K tokens (current pricing)
- ROI timeline: 18-24 months for enterprise deployment

6. LIMITATIONS & OPEN PROBLEMS

Interpretability:
- Attention visualization limited in understanding
- Mechanistic interpretability: Emerging field
- Challenge: Explain 100B+ parameters
- Impact: Critical for healthcare, finance applications

Hallucination & Reliability:
- False information generation: 15-30% of responses
- Lack of uncertainty quantification
- Solution research: Retrieval-augmented generation (RAG)
- Benchmark: TruthfulQA score 42.3%

Computational Cost:
- Training environmental impact: ~630 tons CO2 (GPT-3)
- Inference at scale: Expensive for real-time applications
- Solution: Efficiency research showing promise
- Target: 10x reduction in computational requirements

Context Length Limitations:
- Standard: 4K tokens (GPT-3)
- Extended: 100K tokens (Claude 2)
- Challenge: Longer context = higher memory/compute
- Research direction: Efficient long-context mechanisms

7. FUTURE RESEARCH DIRECTIONS

Mixture of Experts (MoE):
- Sparse activation: Only use relevant experts
- Scaling: 1.6T parameters with same compute as 370B dense
- Advantage: Efficient scaling path
- Challenge: Training instability, load balancing

Retrieval-Augmented Generation (RAG):
- Combine generation with external knowledge
- Reduces hallucinations: 25% improvement on factual tasks
- Enables knowledge updates without retraining
- Emerging standard: 60%+ of new systems use RAG

Multimodal Fusion:
- Beyond concatenation: Learn joint representations
- Applications: 3D scene understanding, embodied AI
- Challenge: Heterogeneous data integration
- Potential: 5-10x improvement over unimodal

Neurosymbolic AI:
- Combine neural networks with symbolic reasoning
- Goal: Interpretability + learning capability
- Applications: Scientific discovery, reasoning tasks
- Status: Early-stage, promising results

8. CONCLUSIONS & IMPLICATIONS

Key Findings:
- Transformers remain dominant architecture (95% of SOTA)
- Scaling is effective but reaching diminishing returns
- Efficiency improvements critical for deployment
- Multimodal learning opening new application domains

Research Priorities:
1. Efficiency & sustainable AI (80% of papers)
2. Reasoning & interpretability (65% of papers)
3. Multimodal understanding (72% of papers)
4. Long-context modeling (45% of papers)

Industry Impact:
- Foundation models becoming standard
- Few-shot learning reducing data requirements
- Cost barriers lowering for smaller orgs
- Commoditization of core capabilities anticipated

References: 150+ citations across 2023-2024 publications
"""

def create_test_files():
    os.makedirs("data/test_docs", exist_ok=True)
    
    files = {
        "data/test_docs/neural_networks.txt": TEST_DOC_1,
        "data/test_docs/knowledge_management.txt": TEST_DOC_2,
        "data/test_docs/research_paper.txt": TEST_DOC_3,
    }
    
    for filepath, content in files.items():
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print("✓ Rich test files created")
    return list(files.keys())