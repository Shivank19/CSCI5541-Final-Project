# Data Curation: SEC Financial Restatement NLP Dataset

## Overview

The goal of this data curation phase was to construct a labeled dataset of earnings call transcripts paired with binary labels indicating whether the company subsequently filed an SEC Item 4.02 restatement notice within a defined time window.

**Final dataset:** 198 examples: 66 positives (label=1) + 132 controls (label=0)  
**Pipeline:** 5 phases implemented in `data.py`

---

## Phase 1: SEC Restatement Collection

### Goal

Identify all 8-K filings with **Item 4.02** (non-reliance on prior financial statements / restatement) from 2015–2024.

### Approach 1: `form.idx` parsing

The original pipeline downloaded quarterly `form.idx` index files from EDGAR and parsed them with fixed-width column offsets to find 8-K filings. This was extremely slow (~5 days runtime) due to the volume of index files and sequential HTTP requests.

**NOTE**: Enter project name and contact email in user agent when using SEC API as per the SEC guidelines

### Approach 2: SEC `submissions` API

Refactored the phase 1 script to query `data.sec.gov/submissions/CIK{cik}.json` for every known CIK. This API returns a structured JSON with the `items` field for each filing, making it trivial to filter for `4.02` without downloading individual filing texts. Used `ThreadPoolExecutor` (12 workers) with a thread-safe token-bucket rate limiter (≤7 req/s) to comply with SEC politeness guidelines.

### Approach 3: Using the SEC provided `submissions.zip` bulk file (**final approach**)

The SEC publishes a bulk `submissions.zip` (~1.5 GB) containing all company submission JSON files. Added `--submissions-zip` CLI argument to scan this file entirely offline, requiring zero API calls. Scanned ~960K JSON files in under 2 minutes.

**Result:** 1,162 raw Item 4.02 records → **1,159 unique restatement filings** after deduplication.

| Field          | Description                       |
| -------------- | --------------------------------- |
| `ticker`       | Stock ticker symbol               |
| `company_name` | Company name from SEC submissions |
| `cik`          | SEC CIK identifier                |
| `filed_at`     | Date of 8-K filing                |
| `sic`          | SIC industry code                 |

**Date range:** 2015-01-09 → 2024-12-27  
**Unique tickers:** 827  
**With SIC codes:** 1,158 / 1,159

---

## Phase 2: Transcript Corpus Assembly

### Goal

Build a pool of earnings call transcripts that can be matched to restatement filings by ticker symbol and date.

### Source 1: MAEC Dataset

~3,400 earnings call transcripts from ~1,188 S&P 500 companies (2015–2020), organized as folders named `{date}_{ticker}`. Each folder contains `text.txt` (full transcript) and `features.csv` (risk labels). Loaded via `--maec <path>` argument. Applied heuristic segmentation to split transcripts into scripted remarks vs. Q&A sections.

### Source 2: Motley Fool Scraped Transcripts ([Kaggle](https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts))

18,755 earnings call transcripts in a `.pkl` file.
Schema: `date`, `exchange`, `quarter`, `ticker`, `full transcript`.
Date column format was `"Aug 27, 2020, 9:00 p.m. ET"` and required regex stripping before `pd.to_datetime` could parse it. Primarily covers data 2019 onwards. Loaded via `--motleyfool <path>` argument.

### Source 3: S&P 500 Earnings Transcripts (HuggingFace `Bose345/sp500_earnings_transcripts`)

~38,000 transcripts from 2005–2025, continuous annual coverage of S&P 500. Schema: `symbol`, `company_name`, `date`, `content`, `structured_content` (speaker/text array). Used `structured_content` for cleaner scripted/Q&A segmentation rather than the regex heuristic. Loaded via `--sp500` flag (auto-downloads from HuggingFace).

### Combined Pool

```
Phase 2 output: data/transcripts_pool.csv
Sources: MAEC + MotleyFool (via Kaggle) + SP500_HF
```

---

## Phase 3: Joining Restatements to Transcripts (Positive Class)

### Goal

For each restatement filing, find an earnings call transcript from the same ticker within a "clean window" before the filing date.

### Clean Window Strategy

Based on Larcker & Zakolyukina (2012): transcripts from the quarter before a restatement are most likely to contain deceptive language by executives who were aware of problems. However, we extended the window to 0-365 days due to lack of data available.

| Window                                     | Matches        |
| ------------------------------------------ | -------------- |
| Original: 90–180 days                      | 4              |
| Widened: 0–365 days                        | 15             |
| + Ticker normalization (strip `.` and `-`) | 15 (no change) |
| + Motley Fool added                        | 66             |
| + S&P 500 HuggingFace added                | 66 (no change) |

### Key Finding: Dataset Mismatch

The fundamental problem is a **company category mismatch**:

- Transcript corpora (MAEC, Motley Fool, S&P 500 HF) skew heavily toward **large/mid-cap** companies with active analyst coverage.
- Item 4.02 restatements disproportionately occur at **small/mid-cap** companies with weaker internal controls, which are not covered by any transcript corpus.
- Of 827 restating tickers, only ~66 appeared in any of the three transcript sources.

Approaches tried to improve match rate:

- **Ticker normalization** (`BRK.B` → `BRKB`): no improvement (formats were already consistent)
- **Window widening** (90–180 → 0–365 days): 4 → 15 matches on MAEC alone
- **Adding Motley Fool** (18K+ transcripts): 15 → 66 matches
- **Adding S&P 500 HF** (38K+ transcripts): 66 → 66 (no new restating companies covered)
- **finpie library** (scrapes Motley Fool live): same source, would not add coverage
- **S&P 500 HF extends 2020–2024 coverage** but restatement companies still not in it

**Final decision:** Accept 66 positives. We acknowledge the limitation of available data.

### Output: `data/positives.csv`

66 rows, label=1, with fields: `ticker`, `call_date`, `filed_at`, `days_before`, `full_text`, `scripted_text`, `qa_text`, `source`, `sic`, `label`

---

## Phase 4: Control Group Assembly

### Goal

For each positive example, select `CONTROL_RATIO (=2)` control transcripts from **non-restating companies** matched on:

1. Same calendar quarter as the positive transcript
2. Same 2-digit SIC code where available (fallback: same year)

Enforced per-company uniqueness, i.e., once a ticker is used as a control, it's excluded from further control selection.

**Result:** 132 control transcripts from 128 unique tickers (label=0)

---

## Phase 5: Final Dataset Assembly & Splits

### Goal

Combine positives and controls, create **company-disjoint temporal splits** to prevent data leakage.

### Split Strategy

Companies ordered chronologically by earliest call date, then divided 70/15/15:

- **Train:** earliest companies, the model trains on older calls
- **Val/Test:** more recent companies, simulates real deployment

No company appears in more than one split (verified with set intersection check).

### Output Files

```
1. restatements.csv: 1,159 SEC Item 4.02 filings
2. transcripts_pool.csv: All loaded transcripts (MAEC + MF + SP500)
3. positives.csv: 66 matched positive examples
4. dataset_final.csv: 198 total (66 pos + 132 ctrl)
5. pipeline_report.txt: Quality report with stats and citations
6. restatements_no_transcript.txt: 1,093 unmatched restating tickers
7. splits/
    7.1 train.csv: ~138 rows (70%)
    7.2 val.csv: ~30 rows (15%)
    7.3 test.csv: ~30 rows (15%)
```

---

## Dataset Limitations

1. **Coverage gap:** Only 8% of restating companies (66/827) have any available earnings call transcript. Small-cap companies are systematically underrepresented in all public transcript corpora.
2. **Source skew:** All three transcript sources (MAEC, Motley Fool, S&P 500 HF) overwhelmingly cover S&P 500 / large-cap companies.
3. **Small dataset:** 198 examples is sufficient for proof-of-concept experiments but not for training large models from scratch. Suitable for fine-tuning or feature-based classifiers.
4. **Control matching:** Matched on quarter + SIC only. Ideally would also match on market cap and profitability metrics (requires paid data source).
5. **Date window:** The 0–365 day window is broader than the 90–180 day "clean window" from prior literature. Some matches may have weaker signal.

---

## How to Run

```bash
# Phase 1: Collect restatements (offline mode, fastest)
python data.py --phase 1 --submissions-zip "data/submissions.zip"

# Phase 2: Load transcript corpora
python data.py --phase 2 \
  --maec "data/MAEC_Dataset" \
  --motleyfool "data/motley-fool-data.pkl" \
  --sp500

# Phase 3–5: Join, build controls, assemble final dataset
python data.py --phase 3
python data.py --phase 4
python data.py --phase 5
```

### Environment Variables

| Variable                    | Default  | Description                 |
| --------------------------- | -------- | --------------------------- |
| `DATA_DIR`                  | `./data` | Output directory            |
| `CLEAN_WINDOW_MIN_DAYS`     | `0`      | Min days before restatement |
| `CLEAN_WINDOW_MAX_DAYS`     | `365`    | Max days before restatement |
| `POSITIVE_TO_CONTROL_RATIO` | `2`      | Controls per positive       |
