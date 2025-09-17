# Misinformation-Detector-Fact-Checking-Agent

Team E:
- Gagan Venkatesh
- Manoj Kumar Pasupuleti
- Swarali Tannu

## Project Overview
This project aims to build a GenAI-powered misinformation detection system that goes beyond simple classification. The system:
- Classifies claims as True, False, Misleading, etc.
- Provides evidence-based explanations via retrieval-augmented verification.
- Offers an API endpoint for real-time fact-checking of text inputs.
Research Question:
- How can AI detect misinformation reliably and provide evidence-based explanations in real time?

## Datasets
1. FACTors 
- ~118,000 fact-checked claims from 1995–2025
- Attributes: Claim, Report ID, Date, Author, Organisation, Verdict, Normalised rating
- Use: Supervised classification, temporal generalisation, fairness analysis

2. FEVER
- ~185,000 claims derived from Wikipedia
- Attributes: Claim text, Label (Supports/Refutes/Not Enough Info), Evidence sentences
- Use: Retrieval-augmented verification and explanation generation
