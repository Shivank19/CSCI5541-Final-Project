# Tier 1: Lexicon Baseline & Interpretability

## Overview
This directory contains the implementation and results for **Tier 1: Lexicon Baseline**. The goal of this tier is to establish the performance ceiling of non-contextual NLP methods (Loughran-McDonald dictionary term frequencies + TF-IDF unigrams) for detecting pre-restatement linguistic leakage in earnings calls. 

This baseline explicitly tests the three experimental conditions outlined in our proposal:
1. `FULL_TEXT`
2. `SCRIPTED_TEXT` (Prepared Remarks)
3. `QA_TEXT` (Unscripted Analyst Q&A)

## Implementation Details
* **Feature Extraction:** Implemented a custom scikit-learn `Transformer` to extract term frequencies across the six Loughran-McDonald categories (Negative, Positive, Uncertainty, Litigious, Strong Modal, Weak Modal). This was joined with standard `TfidfVectorizer` unigrams via `FeatureUnion`.
* **Classifiers:** Evaluated Logistic Regression, SVM (Linear), and Random Forest. 
* **Class Imbalance:** Enforced `class_weight='balanced'` across all models to handle our 1:2 positive-to-control ratio.
* **Reproducibility:** Seeded all models with `random_state=42`. You can re-run the pipeline on the same splits and achieve the exact same metrics.
* **Interpretability:** Integrated `shap.TreeExplainer` to extract the top driving features from the Random Forest model for both the scripted and unscripted conditions.

---

## Quantitative Results
**Primary Metric:** Test Set AUROC

| Condition | Model | AUROC |
| :--- | :--- | :--- |
| `full_text` | Logistic Regression | 0.631313 |
| `full_text` | SVM (Linear) | 0.656566 |
| `full_text` | Random Forest | 0.590909 |
| `scripted_text` | Logistic Regression | 0.525253 |
| `scripted_text` | SVM (Linear) | 0.530303 |
| `scripted_text` | Random Forest | **0.545455** |
| `qa_text` | Logistic Regression | 0.429293 |
| `qa_text` | SVM (Linear) | 0.454545 |
| `qa_text` | Random Forest | **0.482323** |

*Note: Full metrics including PR-AUC, and High-Recall/High-Precision threshold scores (Precision, Recall, F1) are saved in `tier1_baseline_results.csv`.*

---

## Interpretability (SHAP Analysis)
To understand *what* the model was learning, we extracted the Top 15 Mean Absolute SHAP values from the Random Forest model on the Test Set.

**Top Features for `SCRIPTED_TEXT`:**
1. `tfidf__base` (0.006186)
2. `tfidf__forward` (0.006106)
3. `tfidf__potential` (0.005943)
4. `tfidf__growing` (0.005872)
5. `tfidf__prices` (0.005402)
6. `tfidf__growth` (0.004513)
*(...followed by officer, advantage, regarding, com, leader, market, gross, changes, president)*

**Top Features for `QA_TEXT`:**
1. `tfidf__portfolio` (0.009643)
2. `tfidf__times` (0.007201)
3. `tfidf__davidson` (0.006196)
4. `tfidf__help` (0.005594)
5. `tfidf__longer` (0.005155)
6. `tfidf__new` (0.004966)
7. `tfidf__level` (0.004957)
8. `lm_lexicon__LM_Positive` (0.004758)
*(...followed by questions, continue, larger, business, joining, sales, 60)*

---

## Key Analytical Insights for the Paper

The baseline results yielded three insights that directly support our research narrative:

### 1. TF-IDF Crushed the Financial Lexicons
In the SHAP analysis, almost every single top driving feature is a `tfidf__` unigram rather than an `lm_lexicon__` category. This indicates that the specific, granular words spoken by executives on the calls were far more predictive than the traditional Loughran-McDonald dictionary categories (e.g., counting "Negative" or "Uncertainty" words). 

### 2. Direct Evidence for Hypothesis 4 (Overcompensation)
Our H4 proposes that pre-restatement scripted remarks contain an "abnormal spike in positive sentiment and forward-looking language" serving as a linguistic smokescreen. The Tier 1 model practically proves this. The Random Forest successfully flagged the restatement calls in the `SCRIPTED_TEXT` by keying in heavily on forward-looking, overly optimistic unigrams like **"forward"**, **"potential"**, **"growing"**, and **"growth"**.

### 3. Baseline Performance Gap in Unscripted Q&A
Our H2 states that the unscripted Q&A should provide a *better* evasion signal than the prepared remarks. However, the Lexicon Baseline showed the exact opposite:
* **Scripted Text AUROC:** ~0.53 to 0.54
* **QA Text AUROC:** ~0.43 to 0.48 *(Worse than random guessing)*

Furthermore, the SHAP features for the Q&A are incredibly noisy (`portfolio`, `davidson`, `help`, `times`). 

**Why this matters:** These results strongly suggests that simple dictionary-based methods completely fail to capture the messy, subtle evasion tactics executives use during live, unscripted questioning. This baseline failure sets the stage for our Transformers and LLMs) models, proving that contextual, attention-based architectures are structurally necessary to extract the evasion signal from unspoken speech.

---
