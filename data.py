import os
import re
import sys
import json
import time
import random
import zipfile
import argparse
import threading
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")
WINDOW_MIN = int(os.getenv("CLEAN_WINDOW_MIN_DAYS", 0))
WINDOW_MAX = int(os.getenv("CLEAN_WINDOW_MAX_DAYS", 365))
CONTROL_RATIO = int(os.getenv("POSITIVE_TO_CONTROL_RATIO", 2))
START_YEAR = 2015
END_YEAR = 2024
SEED = 42

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "splits"), exist_ok=True)

np.random.seed(SEED)
random.seed(SEED)


# PHASE 1: Restatement Labels

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SUBMISSIONS_ARCHIVE = "https://data.sec.gov/submissions/{filename}"

#Setup headers for SEC API requests
EDGAR_HEADERS = {
    "User-Agent": "project_name contact_email",
    "Accept-Encoding": "gzip, deflate",
}

SEC_MAX_RPS = 7
_rate_lock = threading.Lock()
_req_times = []

def _rate_limited_sec_get(url: str, retries: int = 3):
    while True:
        with _rate_lock:
            now = time.monotonic()
            _req_times[:] = [t for t in _req_times if now - t < 1.0]
            if(len(_req_times) < SEC_MAX_RPS):
                _req_times.append(now)
                break
        time.sleep(0.05)

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=EDGAR_HEADERS, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code == 429:
                print("\n WARNING: SEC rat elimit hit (429), Sleeping 15 seconds before retrying...")
                time.sleep(15)
            elif code == 404:
                return None
            else:
                time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None

def fetch_ticker_map() -> dict:
    data = _rate_limited_sec_get(TICKERS_URL)
    if not data:
        return {}
    return {
        str(v["cik_str"]).zfill(10): {
            "ticker": v["ticker"].upper(),
            "title": v["title"],
        }
        for v in data.values()
    }

def _fetch_company_restatements(cik: str, ticker: str, company_name: str) -> list[dict]:
    url = SUBMISSIONS_URL.format(cik = cik)
    data = _rate_limited_sec_get(url)
    if not data:
        return []
    
    sic = str(data.get("sic", ""))
    name = data.get("name", company_name) or company_name

    # Accumulate all records
    all_forms, all_items, all_dates = [], [], []

    def _extend(block: dict):
        all_forms.extend(block.get("form", []))
        all_items.extend(block.get("items", []))
        all_dates.extend(block.get("filingDate", []))

    _extend(data.get("filings", {}).get("recent", {}))

    for archive_ref in data.get("filings", {}).get("files", []):
        fname = archive_ref.get("name", "")
        if not fname:
            continue
        archive_data = _rate_limited_sec_get(SUBMISSIONS_ARCHIVE.format(filename=fname))
        if archive_data:
            _extend(archive_data)
    
    # Filter to Item 4.02 of 8-K filings
    records = []
    for form, items, date in zip(all_forms, all_items, all_dates):
        if not str(form).startswith("8-K"):
            continue
        if "4.02" not in str(items):
            continue
        if not date or not(str(START_YEAR) <= str(date)[:4] <= str(END_YEAR)):
            continue
        records.append({
            "ticker": ticker,
            "company_name": name,
            "cik": cik,
            "filed_at": date,
            "sic": sic,
        })
    
    return records
    

def _phase1_from_zip(zip_path: str) -> pd.DataFrame:
    
    print(f"\nReading submissions from local zip: {zip_path}")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"submissions.zip not found at: {zip_path}\n"
            "Download from: https://www.sec.gov/Archives/edgar/daily-index/"
            "bulkdata/submissions.zip"
        )
    
    all_records = []

    archive_index: dict[str, list[str]] = {}
    main_files: list[str] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

        for name in names:
            basename = os.path.basename(name)
            if re.match(r"^CIK\d{10}\.json$", basename, re.IGNORECASE):
                main_files.append(name)
            elif re.match(r"^CIK(\d{10})-submissions-\d+\.json$", basename, re.IGNORECASE):
                m = re.match(r"^CIK(\d{10})-", basename, re.IGNORECASE)
                if m:
                    cik = m.group(1).zfill(10)
                    archive_index.setdefault(cik, []).append(name)

        print(f"\nFound {len(main_files):,} company files in zip")

        def _extract_records(filing_block: dict, ticker: str, name: str, cik: str, sic: str) -> list[dict]:
            records = []
            for form, items, date in zip(
                filing_block.get("form", []),
                filing_block.get("items", []),
                filing_block.get("filingDate", [])
            ):
                if not str(form).startswith("8-K"):
                    continue
                if "4.02" not in str(items):
                    continue
                if not date or not(str(START_YEAR) <= str(date)[:4] <= str(END_YEAR)):
                    continue
                records.append({
                    "ticker": ticker,
                    "company_name": name,
                    "cik": cik,
                    "filed_at": date,
                    "sic": sic,
                })
            return records
        
        for fname in tqdm(main_files, desc="Scanning comapny files", unit="co"):
            try:
                with zf.open(fname) as f:
                    data = json.loads(f.read())
            except Exception:
                continue

            tickers_list = data.get("tickers", [])
            ticker = tickers_list[0].upper() if tickers_list else ""
            if not ticker:
                continue

            company_name = data.get("name", "")
            cik = str(data.get("cik", "")).zfill(10)
            sic = str(data.get("sic", ""))

            recent = data.get("filings", {}).get("recent", {})
            all_records.extend(_extract_records(recent, ticker, company_name, cik, sic))

            for archive_fname in archive_index.get(cik, []):
                try:
                    with zf.open(archive_fname) as af:
                        archive_block = json.loads(af.read())
                    all_records.extend(_extract_records(archive_block, ticker, company_name, cik, sic))
                except Exception:
                    continue
    
    print(f"\nRaw Item 4.02 records found: {len(all_records):,}")

    if not all_records:
        raise RuntimeError(
            "No Item 4.02 filings found in zip. "
            "Verify the zip is the SEC submissions bulk file and covers 2015-2024."
        )

    df = pd.DataFrame(all_records)
    df["filed_at"] = pd.to_datetime(df["filed_at"], errors="coerce")
    df = df.dropna(subset=["filed_at"])
    df = df.drop_duplicates(subset=["ticker", "filed_at"])
    df = df.sort_values("filed_at").reset_index(drop=True)

    out = os.path.join(DATA_DIR, "restatements.csv") 
    df.to_csv(out, index=False)

    print(f"{len(df):,} unique restatement filings saved → {out}")
    print(f"Unique tickers: {df['ticker'].nunique():,}")
    print(f"With SIC codes: {(df['sic'] != '').sum():,} / {len(df):,}")
    print(f"Date range: {df['filed_at'].min().date()} → {df['filed_at'].max().date()}")

    return df

def phase1_pull_restatements(zip_path: str = None) -> pd.DataFrame:
    print("\n" + "═"*70)
    print("  PHASE 1 — Pulling restatement labels from SEC EDGAR")
    if zip_path:
        print("(Local Mode: reading local submissions.zip)")
    else:
        print("(Live API mode: concurrent submissions API)")
    print("═"*70)

    if zip_path:
        return _phase1_from_zip(zip_path)
    
    print("\n Loai=ding SEC CIK -> Ticker mapping")
    ticker_map = fetch_ticker_map()
    n_companies = len(ticker_map)
    print(f"Found {n_companies:,} companies in SEC ticker mapping")

    est_min = n_companies // SEC_MAX_RPS // 60
    est_max = int(n_companies * 1.3) // SEC_MAX_RPS // 60
    print(f"\n Estimated time to pull all submissions: {est_min} → {est_max} minutes")

    all_records = []

    with ThreadPoolExecutor(max_workers = 12) as executor:
        futures = {
            executor.submit(
                _fetch_company_restatements,
                cik, info["ticker"], info["title"]
            ): cik
            for cik, info in ticker_map.items()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Pulling Companies", unit="co"):
            try:
                records = future.result()
                if records:
                    all_records.extend(records)
            except Exception:
                pass
    print(f"\nRaw Item 4.02 records found: {len(all_records):,}")

    if not all_records:
        raise RuntimeError(
            "No Item 4.02 filings found via API. "
            "Verify your network connection and SEC API access."
        )
    
    df = pd.DataFrame(all_records)
    df["filed_at"] = pd.to_datetime(df["filed_at"], errors="coerce")
    df = df.dropna(subset=["filed_at"])
    df = df.drop_duplicates(subset=["ticker", "filed_at"])
    df = df.sort_values("filed_at").reset_index(drop=True)

    out = os.path.join(DATA_DIR, "restatements.csv")
    df.to_csv(out, index=False)

    print(f"{len(df):,} unique restatement filings saved → {out}")
    print(f"Unique tickers: {df['ticker'].nunique():,}")
    print(f"With SIC codes: {(df['sic'] != '').sum():,} / {len(df):,}")
    print(f"Date range: {df['filed_at'].min().date()} → {df['filed_at'].max().date()}")

    return df


#PHASE 2: Transcript Corpora (MAEC + Motley Fool via Kaggle + S&P500 ECT via HuggingFace)

QA_BOUNDARY_PATTERNS = [
    r"(?i)we\s+will\s+now\s+(begin|open|take|start).{0,40}question",
    r"(?i)open\s+(the\s+)?(floor|call|line).{0,30}question",
    r"(?i)question[\s\-]+and[\s\-]+answer\s+session",
    r"(?i)\bq\s*[&and]+\s*a\b.{0,20}(session|portion|part)",
    r"(?i)our\s+first\s+question\s+(comes?\s+)?from",
]

ANALYST_PATTERNS = [
    r"(?i)^operator",
    r"(?i)^moderator",
    r"(?i)analyst",
    r"(?i)(research|capital|securities|partners|bank|advisors)",
]


def is_analyst(speaker: str) -> bool:
    return any(re.search(p, speaker.strip()) for p in ANALYST_PATTERNS)


def segment_full_text(full_text: str) -> tuple[str, str]:
    boundary = -1
    for pattern in QA_BOUNDARY_PATTERNS:
        m = re.search(pattern, full_text)
        if m:
            boundary = m.start()
            break

    if boundary == -1:
        mid = len(full_text) // 2
        return full_text[:mid].strip(), full_text[mid:].strip()

    scripted = full_text[:boundary].strip()
    qa_raw   = full_text[boundary:].strip()

    paragraphs = re.split(r"\n{2,}", qa_raw)
    exec_turns = []
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 30:
            continue
        if para.endswith("?") and len(para) < 300:
            continue
        if is_analyst(para.split("\n")[0]):
            continue
        exec_turns.append(para)

    return scripted, "\n\n".join(exec_turns)

def parse_maec_folder_name(folder_name: str) -> tuple[str, str]:
    # Parse folder name (20150225_LMAT => call_date = 2015-02-25, ticker = LMAT)
    m = re.match(r"^(\d{4})(\d{2})(\d{2})_([A-Z0-9\.]+)$", folder_name.upper())

    if not m:
        return "", ""
    year, month, day, ticker = m.groups()
    date_str = f"{year}-{month}-{day}"
    try:
        pd.to_datetime(date_str)
    except Exception:
        return "", ""
    return date_str, ticker

def parse_maec_text(raw_text: str) -> tuple[str, str, str]:
    # Parse raw text into full text vs scripted vs Q&A segments
    full_text = raw_text.strip()
    if not full_text:
        return "", "", ""
    scripted_text, qa_text = segment_full_text(full_text)
    return full_text, scripted_text, qa_text


def load_maec(maec_path: str) -> pd.DataFrame | None:
    # Load MAEC dataset from its GitHub folder structure
    # MAEC folder structure:
    #   MAEC_Dataset/
    #     20150225_LMAT/   ← YearMonthDay_Ticker
    #       text.txt
    #       features.csv
    #     ...
    if not os.path.exists(maec_path):
        print(f"Warning: MAEC path not found: {maec_path}")
        print("Pass the MAEC_Dataset/ folder with --maec")
        return None

    maec_root = Path(maec_path)

    all_folders = [
        f for f in maec_root.iterdir()
        if f.is_dir() and re.match(r"^\d{8}_[A-Za-z0-9\.]+$", f.name)
    ]

    if not all_folders:
        for sub in maec_root.iterdir():
            if sub.is_dir():
                candidate_folders = [
                    f for f in sub.iterdir()
                    if f.is_dir() and re.match(r"^\d{8}_[A-Za-z0-9\.]+$", f.name)
                ]
                if candidate_folders:
                    all_folders = candidate_folders
                    print(f"Found MAEC folders in subdirectory: {sub}")
                    break
    if not all_folders:
        print(f"Warning: No valid MAEC folders found in {maec_path}")
        print("Ensure the structure is correct with folders named like YYYYMMDD_TICKER")
        return None
    
    print(f"Found {len(all_folders):,} MAEC folders to process")

    records, skipped = [], 0

    for folder in tqdm(sorted(all_folders), desc="Processing MAEC folders"):
        call_date, ticker = parse_maec_folder_name(folder.name)
        if not call_date or not ticker:
            skipped += 1
            continue

        txt_path = folder / "text.txt"
        if not txt_path.exists():
            skipped += 1
            continue
        
        try:
            raw = txt_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            skipped += 1
            continue

        if len(raw.strip()) < 200:
            skipped += 1
            continue

        full_text, scripted_text, qa_text = parse_maec_text(raw)
        records.append({
            "ticker": ticker,
            "company_name": "",
            "call_date": call_date,
            "full_text": full_text,
            "scripted_text": scripted_text,
            "qa_text": qa_text,
            "source": "MAEC",
        })

    if skipped > 0:
        print(f"Skipped {skipped} folders due to parsing issues")
    
    if not records:
        print("Warning: No valid MAEC transcripts parsed. Check folder structure and text files.")
        return None
    
    df = pd.DataFrame(records)
    df["company_name"] = ""

    print(f"MAEC: {len(df):,} transcripts loaded")
    print(f"\nDate range: {df['call_date'].min()} → {df['call_date'].max()}")
    print(f"\nUnique tickers: {df['ticker'].nunique():,}")
    n_qa = (df["qa_text"].str.len() > 100).sum()
    print(f"\nTranscripts with Q&A segment: {n_qa:,} / {len(df):,}")

    return df[["ticker", "company_name", "call_date",
               "full_text", "scripted_text", "qa_text", "source"]]

def load_motley_fool(pkl_path: str) -> pd.DataFrame | None:
    
    if not os.path.exists(pkl_path):
        print(f"Warning: Motley Fool pkl file not found: {pkl_path}")
        return None

    print(f"\nLoading Motley Fool transcripts from pickle: {pkl_path}")
    try:
        df = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"Error loading Motley Fool pkl: {e}")
        return None


    print(f"\nLoaded Motley Fool: {len(df):,} rows | columns: {list(df.columns)}")
    print(f"\nRaw 'date' column sample: {df['date'].head(5).tolist()}")
    print(f"\nRaw 'q' column sample: {df['q'].head(5).tolist()}")

    col_map = {}
    for target, candidates in [
        ("ticker",     ["ticker", "symbol", "Ticker", "Symbol"]),
        ("call_date",  ["date", "Date", "call_date", "earnings_date"]),
        ("full_text",  ["transcript", "text", "Transcript", "body"]),
    ]:
        found = next((c for c in candidates if c in df.columns), None)
        if found:
            col_map[found] = target
        else:
            print(f"\nWarning: No column found for {target} in Motley Fool")
            print(f"\nColumns: {list(df.columns)}")
            return None
        
    # Rename columns based on the mapping
    df = df.rename(columns=col_map)

    # Clean and filter
    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["call_date"] = pd.to_datetime(
        df["call_date"].str.replace(r",\s*\d+:\d+.*$", "", regex=True).str.strip(),
        format="%b %d, %Y",
        errors="coerce"
    )

    df = df.dropna(subset=["call_date", "ticker"])
    df = df[df["full_text"].str.len() > 300]

    # Filter to Date Range
    df = df[
        (df["call_date"].dt.year >= START_YEAR) &
        (df["call_date"].dt.year <= END_YEAR)
    ].copy()

    print(f"\n After filtering by date, ({START_YEAR}–{END_YEAR}): {len(df):,} rows remain")
    print(f"\ncall_date sample: {df['call_date'].head(3).tolist() if len(df) > 0 else 'EMPTY'}")

    # Segment into scripted and Q&A

    print(f"\nSegmenting {len(df):,} Motley Fool transcripts …")
    if len(df) == 0:
        print("\nWarning: No Motley Fool transcripts remain after filtering. "
              "Check that call_date column parses correctly and falls within "
              f"{START_YEAR}–{END_YEAR}.")
        return None

    tqdm.pandas(desc="Segmenting Motley Fool")
    df[["scripted_text", "qa_text"]] = df["full_text"].progress_apply(
        lambda x: pd.Series(segment_full_text(x))
    )

    df["company_name"] = ""
    df["source"] = "MotleyFool"

    print(f"\nMotley Fool: {len(df):,} transcripts loaded")
    print(f"\nDate range: {df['call_date'].min().date()} → {df['call_date'].max().date()}")
    print(f"\nUnique tickers: {df['ticker'].nunique():,}")
    df["call_date"] = df["call_date"].dt.strftime("%Y-%m-%d")
    return df[["ticker", "company_name", "call_date",
               "full_text", "scripted_text", "qa_text", "source"]]


def load_sp500_transcripts() -> pd.DataFrame | None:
    # Load S&P500 ECT transcripts from HuggingFace dataset
    try:
       from datasets import load_dataset
    except ImportError:
        print("Warning: datasets library not found. Install with `pip install datasets`.")
        return None
    
    print("\nLoading S&P500 ECT transcripts from HuggingFace (Bose345/sp500_earnings_transcripts)")
    try:
        ds = load_dataset("Bose345/sp500_earnings_transcripts", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"\nError loading S&P500 ECT dataset: {e}")
        return None

    df = ds.to_pandas()
    print(f"\nLoaded S&P500 ECT: {len(df):,} rows | columns: {list(df.columns)}")

    df = df.rename(columns={"symbol": "ticker", "content": "full_text"})

    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["call_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["call_date", "ticker"])
    df = df[df["full_text"].str.len() > 300]

    # Filter to date range
    df = df[
        (df["call_date"].dt.year >= START_YEAR) &
        (df["call_date"].dt.year <= END_YEAR)
    ].copy()

    print(f"\n After filtering by date, ({START_YEAR}–{END_YEAR}): {len(df):,} rows remain")

    if len(df) == 0:
        print("\nWarning: No S&P500 ECT transcripts remain after filtering. ")
        return None


    def segment_structured(row):
        sc = row.get("structured_content", None)
        full = str(row["full_text"])

        try:
            valid_sc = sc is not None and len(sc) > 0 and isinstance(sc[0], dict)
        except(TypeError, KeyError, IndexError):
            valid_sc = False
        if not valid_sc:
            return pd.Series(segment_full_text(full))
        
        qa_keywords = {"analyst", "question", "operator", "q&a", "q & a"}

        scripted_parts, qa_parts = [], []
        in_qa = False

        for turn in sc:
            speaker = str(turn.get("speaker", "")).lower()
            text = str(turn.get("text", ""))
            if not in_qa and any(k in speaker for k in qa_keywords):
                in_qa = True
            if in_qa:
                qa_parts.append(text)
            else:
                scripted_parts.append(text)
        return pd.Series((" ".join(scripted_parts), " ".join(qa_parts)))
    
    print(f"\nSegmenting {len(df):,} S&P500 transcripts")
    tqdm.pandas(desc="Segmenting S&P500 ECT")
    df[["scripted_text", "qa_text"]] = df.progress_apply(segment_structured, axis=1)

    df["source"] = "SP500_HF"

    print(f"\nS&P500 ECT: {len(df):,} transcripts loaded")
    print(f"\nDate range: {df['call_date'].min().date()} → {df['call_date'].max().date()}")
    print(f"\nUnique tickers: {df['ticker'].nunique():,}")
    return df[["ticker", "company_name", "call_date",
               "full_text", "scripted_text", "qa_text", "source"]]
            
def phase2_load_transcripts(maec_path: str = None, motleyfool_path: str = None, load_sp500: bool = False) -> pd.DataFrame:
    print("\n" + "═"*70)
    print("\nPHASE 2: Loading and parsing earnings call transcripts")
    print("\n═"*70)

    frames = []

    if maec_path:
        df_maec = load_maec(maec_path)
        if df_maec is not None:
            frames.append(df_maec)
            print(f"\nMAEC: {len(df_maec):,} transcripts")
    else:
        print("\nSkipping MAEC (no path provided). Pass --maec <path> to include.")
    if motleyfool_path:
        df_mf = load_motley_fool(motleyfool_path)
        if df_mf is not None:
            frames.append(df_mf)
            print(f"\nMotleyFool: {len(df_mf):,} transcripts")
    else:
        print("\nSkipping Motley Fool (no path provided). Pass --motleyfool <path> to include.")
    if load_sp500:
        df_sp = load_sp500_transcripts()
        if df_sp is not None:
            frames.append(df_sp)
            print(f"\nS&P500 ECT: {len(df_sp):,} transcripts")
    else:
        print("\nSkipping S&P500 ECT (not enabled). Pass --sp500 to include.")

    if not frames:
        raise RuntimeError("No transcripts loaded from any source. Check paths and dataset availability.")
    
    pool = pd.concat(frames, ignore_index=True)
    pool = pool.drop_duplicates(subset=["ticker", "call_date"])
    pool = pool[pool["full_text"].str.len() > 300].reset_index(drop=True)
    pool["full_text"] = pool["full_text"].astype(str)
    pool["scripted_text"] = pool["scripted_text"].astype(str)
    pool["qa_text"] = pool["qa_text"].astype(str)

    pool = pool.reset_index(drop=True)
    out = os.path.join(DATA_DIR, "transcripts_pool.csv")
    pool.to_csv(out, index=False)

    print(f"\nCombined pool: {len(pool):,} transcripts → {out}")
    print(f"\nUnique tickers: {pool['ticker'].nunique():,}")
    print(f"\nSources: {pool['source'].value_counts().to_dict()}")
    n_qa = (pool["qa_text"].str.len() > 100).sum()
    print(f"\nTranscripts with Q&A segment: {n_qa:,} / {len(pool):,}")

    return pool

# PHASE 3: Match restatements to transcripts

def phase3_join_positives(df_rest: pd.DataFrame, df_pool: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═"*70)
    print("  PHASE 3 — Matching restatements to transcripts")
    print("═"*70)
    print(f"  Clean window: {WINDOW_MIN}–{WINDOW_MAX} days before 8-K filing\n")

    df_rest = df_rest.copy()
    df_pool = df_pool.copy()

    # Normalize tickers
    def _norm_ticker(s: pd.Series) -> pd.Series:
        return s.str.upper().str.strip()\
                .str.replace(".", "", regex=False)\
                .str.replace("-", "", regex=False)

    df_rest["ticker"] = _norm_ticker(df_rest["ticker"])
    df_pool["ticker"] = _norm_ticker(df_pool["ticker"])

    overlap = set(df_rest["ticker"]) & set(df_pool["ticker"])
    print(f"Overlapping tickers: {len(overlap):,}")

    df_rest["filed_at"] = pd.to_datetime(df_rest["filed_at"])
    df_pool["call_date"] = pd.to_datetime(df_pool["call_date"], errors="coerce")
    df_pool = df_pool.dropna(subset=["call_date"])

    records, id_counter, no_match = [], 1, []

    for _, rest_row in tqdm(df_rest.iterrows(), total=len(df_rest), desc="Matching"):
        ticker = rest_row["ticker"]
        announce_date = rest_row["filed_at"]
        window_start = announce_date - timedelta(days=WINDOW_MAX)
        window_end = announce_date - timedelta(days=WINDOW_MIN)

        candidates = df_pool[
            (df_pool["ticker"] == ticker) &
            (df_pool["call_date"] >= window_start) &
            (df_pool["call_date"] <= window_end)
        ]

        if len(candidates) == 0:
            no_match.append(ticker)
            continue

        candidates = candidates.copy()
        candidates["days_before"] = (announce_date - candidates["call_date"]).dt.days
        best = candidates.sort_values("days_before").iloc[0]

        records.append({
            "transcript_id": f"T{id_counter:05d}",
            "ticker": ticker,
            "company_name": best.get("company_name", rest_row.get("company_name", "")),
            "call_date": best["call_date"].date().isoformat(),
            "label": 1,
            "days_before": int(best["days_before"]),
            "announce_date": announce_date.date().isoformat(),
            "sic": rest_row.get("sic", ""),
            "source": best.get("source", ""),
            "full_text": best["full_text"],
            "scripted_text": best.get("scripted_text", ""),
            "qa_text": best.get("qa_text", ""),
        })
        id_counter += 1

    df_pos = pd.DataFrame(records)
    out = os.path.join(DATA_DIR, "positives.csv")
    df_pos.to_csv(out, index=False)

    match_rate = len(df_pos) / len(df_rest) * 100 if len(df_rest) > 0 else 0
    print(f"\n Restatement Events: {len(df_rest):,}")
    print(f"\n Matched to Transcripts: {len(df_pos):,}")
    print(f"\n Match Rate: {match_rate:.2f}%")
    print(f"\n Unmatched Tickers: {len(no_match):,} | Examples: {no_match[:5]}")
    if len(df_pos) > 0:
        print(f"\n Average days before restatement: {df_pos['days_before'].mean():.1f}")
    if no_match:
        nf = os.path.join(DATA_DIR, "restatements_no_transcript.txt")
        with open(nf, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(set(no_match))))
        print(f"\nUnmatched tickers saved → {nf}")
    return df_pos

# PHASE 4: Build Control Group
def get_quarter(date) -> str:
    dt = pd.to_datetime(date)
    return f"{dt.year}Q{(dt.month - 1) // 3 + 1}"

def get_sic2(sic) -> str:
    return str(sic).zfill(4)[:2] if sic else "00"

def phase4_build_controls(df_pos: pd.DataFrame, df_pool: pd.DataFrame, df_rest: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═"*70)
    print("\nPHASE 4: Building control group")
    print("═"*70)
    print(f"\nControl ratio- {CONTROL_RATIO}:1 (controls:positives)\n")

    restating_tickers = set(df_rest["ticker"].str.upper())

    
    df_pool = df_pool.copy()
    df_pool["call_date"] = pd.to_datetime(df_pool["call_date"], errors="coerce")
    df_pool["quarter"] = df_pool["call_date"].apply(lambda d: get_quarter(d) if pd.notna(d) else "")
    df_pool["sic2"] = ""

    control_pool = df_pool[~df_pool["ticker"].isin(restating_tickers)].copy()
    \
    print(f"Control candidate pool: {len(control_pool):,} transcripts from {control_pool['ticker'].nunique():,} tickers")

    records, id_counter, used_ids = [], 1, set()

    for _, pos_row in tqdm(df_pos.iterrows(), total=len(df_pos), desc="Matching Controls"):
        target_quarter = get_quarter(pos_row["call_date"])

        candidates = control_pool[ (control_pool["quarter"] == target_quarter) & ~control_pool.index.isin(used_ids)]

        if len(candidates) < CONTROL_RATIO:
            year = target_quarter[:4]
            candidates = control_pool[ (control_pool["quarter"].str.startswith(year)) & ~control_pool.index.isin(used_ids)]

        if len(candidates) == 0:
            continue
        
        sample = (candidates.drop_duplicates(subset=["ticker"]).sample(min(CONTROL_RATIO, len(candidates)), random_state=SEED))

        for _, ctrl_row in sample.iterrows():
            records.append({
                "transcript_id": f"C{id_counter:05d}",
                "ticker": ctrl_row["ticker"],
                "company_name": ctrl_row.get("company_name", ""),
                "call_date": ctrl_row["call_date"].date().isoformat() if pd.notna(ctrl_row["call_date"]) else "",
                "label": 0,
                "days_before": None,
                "announce_date": None,
                "sic": "",
                "source": ctrl_row.get("source", ""),
                "full_text": ctrl_row["full_text"],
                "scripted_text": ctrl_row.get("scripted_text", ""),
                "qa_text": ctrl_row.get("qa_text", ""),
            })
            used_ids.add(ctrl_row.name)
            id_counter += 1

    df_ctrl = pd.DataFrame(records)
    print(f"\n{len(df_ctrl):,} control transcripts matched")
    print(f"\nUnique control tickers: {df_ctrl['ticker'].nunique():,}")
    return df_ctrl

# PHASE 5: Assemble Final Dataset and Create Splits

def phase5_assemble_and_split(df_pos: pd.DataFrame, df_ctrl: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "═"*70)
    print("\nPHASE 5: Assembling final dataset and creating splits")
    print("═"*70)

    df_all = pd.concat([df_pos, df_ctrl], ignore_index=True)
    df_all["call_date"] = pd.to_datetime(df_all["call_date"], errors="coerce")
    df_all = df_all.dropna(subset=["call_date"])
    df_all = df_all.sample(frac=1, random_state=SEED).reset_index(drop=True)

    company_first_date = (df_all.groupby("ticker")["call_date"].min().sort_values().reset_index())
    companies = company_first_date["ticker"].tolist()
    n = len(companies)

    train_companies = set(companies[: int(n * 0.70)])
    val_companies = set(companies[int(n * 0.70) : int(n * 0.85)])
    test_companies = set(companies[int(n * 0.85):])

    df_all["split"] = df_all["ticker"].apply(lambda t: "train" if t in train_companies else ("val" if t in val_companies else "test"))

    final_path = os.path.join(DATA_DIR, "dataset_final.csv")
    df_all.to_csv(final_path, index=False)
    print(f"\nFinal dataset: {len(df_all):,} transcripts → {final_path}")

    for split in ["train", "val", "test"]:
        sub = df_all[df_all["split"] == split]
        path = os.path.join(DATA_DIR, "splits", f"{split}.csv")
        sub.to_csv(path, index=False)
        pos = sub["label"].sum()
        neg = len(sub) - pos
        print(f"  {split:5s}: {len(sub):4d} total  |  {pos} pos / {neg} neg"
              f"  |  {sub['ticker'].nunique()} companies")

    train_t = set(df_all[df_all["split"] == "train"]["ticker"])
    val_t = set(df_all[df_all["split"] == "val"]["ticker"])
    test_t = set(df_all[df_all["split"] == "test"]["ticker"])

    leakage = (train_t & val_t) | (train_t & test_t) | (val_t & test_t)

    if leakage:
        print(f"\nWarning: {len(leakage)} tickers appear in multiple splits")
    else:
        print("\nNo ticker leakage across splits")
    return df_all

# Main
def main():
    parser = argparse.ArgumentParser(description="Lexicon Labs Dataset Pipeline")
    parser.add_argument("--phase", type=int, help="Phase to run (1-5)")
    parser.add_argument("--from-phase", type=int, help="Run from specified phase to end")
    parser.add_argument("--maec", type=str, default=None, help="Path to MAEC_Dataset/ folder (optional argument)")
    parser.add_argument("--motleyfool", type=str, default=None, help="Path to Motley Fool transcripts pickle (optional argument)")
    parser.add_argument("--sp500", action="store_true", help="Whether to load S&P500 EC transcripts from HuggingFace (Bose345/sp500_earnings_transcripts) (optional flag)")
    parser.add_argument("--submissions-zip", type=str, default=None, help="Path to local submissions.zip file for Phase 1(optional argument)")

    args = parser.parse_args()

    run_from = args.from_phase or (args.phase if args.phase else 1)
    run_to = args.phase or 5

    # Phase 1:
    rest_path = os.path.join(DATA_DIR, "restatements.csv")
    if run_from <= 1 <= run_to:
        df_rest = phase1_pull_restatements(zip_path=args.submissions_zip)
    elif os.path.exists(rest_path):
        df_rest = pd.read_csv(rest_path, parse_dates=["filed_at"])
        print(f"\nLoaded restatements from {rest_path} ({len(df_rest):,} records)")
    else:
        print(f"\nrestatements.csv not found at {rest_path}. Run Phase 1 first.")
        sys.exit(1)

    # Phase 2:
    pool_path = os.path.join(DATA_DIR, "transcripts_pool.csv")
    if run_from <= 2 <= run_to:
        df_pool = phase2_load_transcripts(maec_path=args.maec, motleyfool_path=args.motleyfool, load_sp500=args.sp500)
    
    elif os.path.exists(pool_path):
        df_pool = pd.read_csv(pool_path)
        print(f"\nLoaded transcripts pool from {pool_path} ({len(df_pool):,} records)")
    
    else:
        print(f"\ntranscripts_pool.csv not found at {pool_path}. Run Phase 2 first.")
        sys.exit(1)

    # Phase 3:
    pos_path = os.path.join(DATA_DIR, "positives.csv")
    if run_from <= 3 <= run_to:
        df_pos = phase3_join_positives(df_rest, df_pool)
    elif os.path.exists(pos_path):
        df_pos = pd.read_csv(pos_path)
        print(f"\nLoaded existing positives.csv from {pos_path} ({len(df_pos):,} records)")
    else:
        print(f"\npositives.csv not found at {pos_path}. Run Phase 3 first.")
        sys.exit(1)

    if run_to < 4:
        return
    
    # Phase 4:
    ctrl_path = os.path.join(DATA_DIR, "controls.csv")
    if run_from <= 4 <= run_to:
        df_ctrl = phase4_build_controls(df_pos, df_pool, df_rest)
        df_ctrl.to_csv(ctrl_path, index=False)
        print(f"\nControl group saved → {ctrl_path}")
    elif os.path.exists(ctrl_path):
        df_ctrl = pd.read_csv(ctrl_path)
        print(f"\nLoaded existing controls.csv from {ctrl_path} ({len(df_ctrl):,} records)")
    else:
        print(f"\ncontrols.csv not found at {ctrl_path}. Re-runing Phase 4 to build controls.")
        df_ctrl = phase4_build_controls(df_pos, df_pool, df_rest)
        df_ctrl.to_csv(ctrl_path, index=False)

    # Phase 5:
    if run_from <= 5 <= run_to:
        df_all = phase5_assemble_and_split(df_pos, df_ctrl)
        print("\n" + "═"*70)
        print("  PIPELINE COMPLETE")
        print("  Outputs in:", DATA_DIR)
        print("═"*70)