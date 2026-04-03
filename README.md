# Linguistic Precursors to Financial Restatements

**Course:** CSCI 5541: Natural Language Processing, Spring 2026, University of Minnesota  
**Team:** Lexicon Labs
**Task:** Supervised binary classification: _did an earnings call transcript precede a financial restatement?_

## Overview

This project investigates whether corporate executives’ language in earnings call transcripts contains detectable signals that precede financial restatements.

A financial restatement (SEC Form 8-K Item 4.02) indicates that previously reported financial statements are unreliable. We study whether linguistic patterns like hedging, uncertainty, or evasiveness appear in earnings calls before such events are publicly disclosed.

---

## Dataset

#### Approach:

We construct a labeled dataset through a five-phase pipeline:

1. Extract SEC Item 4.02 restatement filings
2. Assemble earnings call transcripts from multiple sources
3. Match transcripts occurring before restatements (positive class)
4. Construct matched control samples (non-restating firms)
5. Build a cleaned, labeled dataset with company-disjoint splits

The pipeline ensures temporal correctness, reproducibility, and realistic handling of data constraints.

- **Total samples:** 198
- **Positive (restatement):** 66
- **Control:** 132
- **Sources:** SEC EDGAR, MAEC, Motley Fool

Due to real-world data availability, only a subset of restating companies had matching transcripts. The project is therefore framed as a **small-data empirical study**.

See [`DATA_CURATION_README.md`](./DATA_CURATION_README.md) for full details on:

- data sources
- pipeline design
- reproducibility

**NOTE**: Enter project name and contact email in user agent when using SEC API as per the SEC guidelines

---

## Modeling Plan

We evaluate three tiers of models:

- **Tier 1:** Lexicon-based baseline (Loughran–McDonald)
- **Tier 2:** Fine-tuned transformer models (e.g., FinBERT)
- **Tier 3:** Zero-/few-shot LLM prompting (e.g., GPT, LLaMA)

---

## Key Challenges

- Limited overlap between restating firms and transcript datasets
- Bias toward large-cap companies in public transcript corpora
- Small dataset size requiring careful evaluation and modeling choices

---

## Reproducibility

The dataset is fully reproducible using the provided pipeline, given access to the original data sources.

See [`DATA_CURATION_README.md`](./DATA_CURATION_README.md) for:

- step-by-step pipeline instructions
- required data sources
- configuration details

---

## Status

- Data pipeline: COMPLETE
- Dataset construction: COMPLETE
- Modeling: IN PROGRESS
- Analysis and evaluation: TBD

---

## Acknowledgements

- SEC EDGAR (U.S. Securities and Exchange Commission)
- MAEC Dataset (CIKM 2020)
- Motley Fool Earnings Call Transcripts (Kaggle)

---
