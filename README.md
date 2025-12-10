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


## Current phase (Phase 2) covers:
- Complete data cleaning and exploratory analysis (EDA)
- Temporal data splits (train ≤ 2022 · val = 2023 · test ≥ 2024)
- Baseline model training using DeBERTa-base
- Model evaluation using Macro-F1, Accuracy, AUROC, and Expected Calibration Error (ECE)
- Runtime optimization (training time reduced ~ 4 h → 1 h)

## Datasets
FACTors [Dataset](https://github.com/altuncu/FACTors)
- ~118,000 fact-checked claims from 1995–2025
- Attributes: Claim, Report ID, Date, Author, Organisation, Verdict, Normalised rating
- Use: Supervised classification, temporal generalisation, fairness analysis

## Data Preparation
- Removed invisible characters / duplicates
- Parsed ISO dates → derived Year & Month
- Normalized text + verdict labels (misleading → partially true)
- Saved clean dataset → reports/tables/FACTors_clean.csv

## Exploratory Data Analysis (EDA)
- Label imbalance: majority False / Partially True
- Temporal trend: rapid growth after 2015 (peaks in 2020 & 2024)
- Authorship bias: few authors/orgs produce most fact-checks
- Claim length: median ≈ 15 words (short factual claims)

## Model Construction (DeBERTa-Base)
- Framework: Hugging Face Transformers + PyTorch
- Model: microsoft/deberta-base
- Token length = 192 · Batch = 16/32 · Epochs = 2 · LR = 4e-5
- Weighted CrossEntropy Loss for class imbalance
- Temporal split ensures future-year generalization


## Runtime Optimizations
- Enabled TF32 on Ampere GPUs:	Faster matrix multiplies
- Dynamic padding + max len 192:	Smaller batches
- Froze bottom 4 layers:	Fewer trainable params
- Reduced epochs 3 → 2:	Shorter training
- Logging steps 200:	Less overhead
<br>Result → Training time cut ~ 4 h → 1 h with negligible accuracy loss.

## Results (Validation 2023)
- Loss	1.3301
- Accuracy	0.6173
- Macro-F1	0.3866
- AUROC (OVR)	0.8090
- ECE (15 bins)	0.0530 → 5.3%

## Next Steps (Phase 3)
- Implement retrieval-augmented verification (FAISS / SBERT)
- Generate LLM-based explanations citing evidence
- Build FastAPI endpoint /analyze?text= for real-time fact-checking

