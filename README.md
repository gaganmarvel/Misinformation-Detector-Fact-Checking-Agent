# Misinformation-Detector-Fact-Checking-Agent

Team E:
- [Gagan Venkatesh](https://github.com/gaganmarvel "Gagan Venkatesh")
- [Manoj Kumar Pasupuleti](https://github.com/ManojUMBC "Manoj Kumar Pasupuleti")
- [Swarali Tannu](https://github.com/swaralitannu "Swarali Tannu")

## Project Overview
This project aims to build a GenAI-powered misinformation detection system that goes beyond simple classification. The system:
- Classifies claims as True, False, Misleading, Partially True, Unverifiable, and Other.
- Provides evidence-based explanations via retrieval-augmented verification.
- Offers an API endpoint for real-time fact-checking of text inputs.
<br>Research Question:
- How can AI detect misinformation reliably and provide evidence-based explanations in real time?

## Datasets
FACTors [Dataset](https://github.com/altuncu/FACTors)
- ~118,000 fact-checked claims from 1995–2025
- Attributes: Claim, Report ID, Date, Author, Organisation, Verdict, Normalised rating
- Use: Supervised classification, temporal generalisation, fairness analysis

## Data Cleaning & Preprocessing
- Removed Unicode noise, zero-width characters.
- Language detection → filtered English-only claims
- Normalized smart quotes, whitespace, and casing
- Fixed missing/short claims using titles
- Merged “Misleading → Partially True”
- Parsed dates → extracted year
- Removed duplicates using (claim, organisation, date)
- Output: FACTors_clean.csv

## Exploratory Data Analysis (EDA)
Key findings:
- Class imbalance: “False” and “Partially True” dominate
- Temporal spikes: Fact-checking volume increases sharply after 2015
- Org dependence: PolitiFact and Snopes provide most fact-checks
- Author skew: Many claims assigned to “Unknown” authors
- Short claims: Most claims < 20 words <br>
These insights informed model choices such as Focal Loss, WeightedRandomSampler, and year embeddings.

## Temporal Data Splits
To simulate real deployment:
- Train	≤ 2022 → Historical misinformation
- Validation = 2023 → Model tuning year
- Test	≥ 2024 → Future misinformation evaluation <br>
This prevents leakage and evaluates true robustness under distribution shift.

## Model Architecture 
- Backbone: DeBERTa-v3-base. Chosen for its disentangled attention + enhanced contextual modeling.
- Year Embedding (32-dim): injected as metadata to help the model learn temporal drift patterns.
- Classifier: Concatenates [CLS] + YearEmbedding  Linear  5-class output.
- Focal Loss (gamma = 2.0): combats severe label imbalance.
- WeightedRandomSampler: oversamples rare labels without distorting distribution too harshly.

## Validation Results
- Accuracy = 0.6728
- Macro-F1 = 0.4058 
- Macro Precision = 0.3982
- Macro Recall = 0.4310
- ECE (Expected Calibration Error) = 0.0187

## Test Set Evaluation 
- Accuracy: 0.6560
- Macro-F1: 0.3519
- Macro Precision: 0.3586
- Macro Recall: 0.3702
- ECE (Expected Calibration Error): 0.0173

## Retrieval-Augmented Verification
- Embedding Model - SBERT: all-MiniLM-L6-v2
- Retrieval Pipeline
-- Encodes all train claims
-- Builds FAISS IndexFlatIP for cosine similarity
-- Retrieves top-k (k=5) past fact-checks for each test claim
- Evaluation
-- Recall@k and MRR
-- Semantic duplicate evaluation using cosine ≥ 0.88 <br>
Retrieval allows the system to cite historical fact-checks highly relevant to the input claim.

## GEN AI Explanations
### Without Web Evidence
- FLAN-T5 generates fact-check explanations using retrieved FACTors claims only
- Ensures low hallucination risk
- Useful for controlled evaluations
### With Web Evidence
- Wikipedia API → fallback to DuckDuckGo
- GPT-4o-mini produces explanations citing:
-- [F1], [F2], … for FACTors
-- [W1], [W2], … for web evidence
### Attribution (NLI Grounding)
- RoBERTa-MNLI validates each generated sentence
- Produces an Attribution Support Rate
- Ensures explanations are grounded and trustworthy

## Fairness & Temporal Evaluation
For each year in test set:
- Accuracy, Macro-F1, Precision, Recall computed
- Clear evidence of performance degradation on newer misinformation, confirming real-world drift
- Highlights need for continual fine-tuning or domain adaptation

## API Deployment (FastAPI)
### Endpoints
POST /analyze
### Input:
{
  "claim": "Vaccines cause autism",
  "year": 2024
}
### Output includes:
- Predicted label
- Class probabilities
- Retrieved FACTors evidence
- Web evidence
- Generated explanation
- NLI-based grounding score
- Confidence decomposition

## Key References
- Wang, W. Y. (2017). “Liar, Liar Pants on Fire”: A new benchmark dataset for fake news detection. ACL. https://arxiv.org/abs/1705.00648 
- Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). FEVER: A large-scale dataset for fact verification. NAACL-HLT. https://arxiv.org/abs/1803.05355
- Altuncu, M. T., et al. (2025). FACTors: A benchmark dataset for fact-checking research. SIGIR. https://arxiv.org/abs/2505.09414 
- He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with disentangled attention. ICLR. https://arxiv.org/abs/2006.03654
- Thorne, J., Vlachos, A. (2018). Automated fact-checking: Task formulations, methods, and future directions. ACL. https://arxiv.org/abs/1806.07687
