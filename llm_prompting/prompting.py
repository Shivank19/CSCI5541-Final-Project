import os
import re
import json
import time
import random
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

################
USE_FINGPT = True
################

warnings.filterwarnings("ignore")


DATA_DIR = "../data"
OUTPUT_DIR = "./evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CONDITIONS = ["qa_text", "scripted_text"]

N_FEWSHOT_POS = 2
N_FEWSHOT_NEG = 2
REQUEST_SLEEP_SEC = 0.0  


# Change this to "llama3:70b" when running on a GPU rig
LLAMA_OLLAMA_MODEL = "llama3"  
LLAMA_MAX_NEW_TOKENS = 160
LLAMA_MAX_CHARS = 25000 

FINGPT_MODEL_NAME = "FinGPT/fingpt-sentiment_llama2-13b_lora"
FINGPT_BASE_MODEL = "NousResearch/Llama-2-13b-hf"
FINGPT_MAX_NEW_TOKENS = 160
DEVICE_MAP = "auto"

FINGPT_MAX_CHARS = 8000 


# =========================
# Llama Prompt Variants
# =========================
PROMPT_VARIANTS: Dict[str, str] = {
    "evasion": """You are a financial risk analyst.
Your task is to judge whether the following earnings-call text shows elevated risk of a future financial restatement.
Focus especially on:
- evasiveness
- indirect or non-responsive answers
- unusual hedging or uncertainty
- defensive or overconfident language

{few_shot_block}

Target text:
{text}

Return ONLY valid JSON in this exact schema:
{{
  "label": 0 or 1,
  "risk_score": integer from 0 to 100,
  "rationale": "one short sentence"
}}
Where:
- label=1 means elevated future restatement risk
- label=0 means no elevated future restatement risk
- risk_score must be higher when restatement risk is higher""",

    "section_aware": """You are a financial language analyst studying pre-restatement linguistic signals.
A risky transcript may show either:
- evasive, vague, or hedged executive language
- unusually upbeat or defensive language that may overcompensate for hidden problems

{few_shot_block}

Target text:
{text}

Return ONLY valid JSON in this exact schema:
{{
  "label": 0 or 1,
  "risk_score": integer from 0 to 100,
  "rationale": "one short sentence"
}}
Interpretation:
- risk_score 0 = very unlikely to precede a restatement
- risk_score 100 = very likely to precede a restatement""",
}


# =========================
# Utility Functions
# =========================
def truncate_text(text: str, max_chars: int) -> str:
    """Dynamically truncates text based on the specific model's context limit."""
    if not isinstance(text, str):
        return ""
    return text.strip()[:max_chars]


def calculate_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["AUROC"] = float(roc_auc_score(y_true, y_score))
    metrics["PR_AUC"] = float(average_precision_score(y_true, y_score))
    metrics["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["Recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["F1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return metrics


def robust_json_parse(raw: str) -> Tuple[Optional[Dict], bool]:
    if not isinstance(raw, str) or not raw.strip():
        return None, False

    cleaned = raw.strip()
    cleaned = re.sub(r"```json|```", "", cleaned, flags=re.IGNORECASE).strip()

    try:
        return json.loads(cleaned), True
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), True
        except Exception:
            return None, False

    return None, False


def parse_model_response(raw: str) -> Dict:
    parsed, parse_ok = robust_json_parse(raw)
    out = {
        "pred_label": -1,
        "risk_score": None,
        "rationale": "",
        "parse_ok": parse_ok,
    }

    if parsed is None:
        return out

    label = parsed.get("label", -1)
    try:
        label = int(label)
    except Exception:
        label = -1
    if label not in [0, 1]:
        label = -1

    risk = parsed.get("risk_score", None)
    try:
        risk = int(risk)
    except Exception:
        risk = None
    if risk is not None:
        risk = max(0, min(100, risk))

    rationale = str(parsed.get("rationale", "")).strip()

    out.update({
        "pred_label": label,
        "risk_score": risk,
        "rationale": rationale,
    })
    return out


# =========================
# Few-Shot Selection
# =========================
def choose_fixed_fewshot_examples(
    train_df: pd.DataFrame,
    condition: str,
    n_pos: int = N_FEWSHOT_POS,
    n_neg: int = N_FEWSHOT_NEG,
    random_state: int = RANDOM_SEED,
) -> List[Dict]:
    rng = np.random.default_rng(random_state)

    pos_df = train_df[(train_df["label"] == 1) & train_df[condition].notna()].copy()
    neg_df = train_df[(train_df["label"] == 0) & train_df[condition].notna()].copy()

    pos_idx = rng.choice(pos_df.index.to_numpy(), size=min(n_pos, len(pos_df)), replace=False)
    neg_idx = rng.choice(neg_df.index.to_numpy(), size=min(n_neg, len(neg_df)), replace=False)

    examples = pd.concat([pos_df.loc[pos_idx], neg_df.loc[neg_idx]], axis=0)
    examples = examples.sample(frac=1.0, random_state=random_state)
    return examples.to_dict("records")


def format_fewshot_block(examples: List[Dict], condition: str, max_chars: int) -> str:
    """Formats few-shot examples with rationales mapped to the specific hypothesis being tested."""
    if not examples:
        return ""

    blocks: List[str] = ["Here are labeled examples:\n"]
    for i, ex in enumerate(examples, start=1):
        ex_text = truncate_text(ex.get(condition, ""), min(1200, max_chars))
        label = int(ex["label"])
        risk_score = 75 if label == 1 else 20
        
        if label == 1:
            if condition == "qa_text":
                rationale = "The executive's unscripted response is evasive, hedged, and lacks directness."
            elif condition == "scripted_text":
                rationale = "The prepared remarks are unusually upbeat and forward-looking, acting as a smokescreen for underlying issues."
            else:
                rationale = "The language exhibits markers of risk or misreporting."
        else:
            rationale = "The language is direct, ordinary, and not especially suspicious."

        blocks.append(
            f"Example {i}:\n"
            f"Text:\n{ex_text}\n\n"
            f"Response:\n"
            f"{{\"label\": {label}, \"risk_score\": {risk_score}, \"rationale\": \"{rationale}\"}}\n"
        )
    return "\n".join(blocks)

@dataclass
class ModelWrapper:
    name: str
    generate_fn: Callable[[str], str]
    max_chars: int
    is_fingpt: bool = False

def build_ollama_llama_wrapper(model_name: str = LLAMA_OLLAMA_MODEL) -> ModelWrapper:
    try:
        import ollama  
    except ImportError as e:
        raise ImportError("Install ollama Python client: pip install ollama") from e

    client = ollama.Client()

    def _generate(prompt: str) -> str:
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.001, 
                "num_predict": LLAMA_MAX_NEW_TOKENS,
            },
            format="json",
        )
        return response["message"]["content"].strip()

    return ModelWrapper(name=f"llama::{model_name}", generate_fn=_generate, max_chars=LLAMA_MAX_CHARS)

def build_fingpt_wrapper(
    lora_model_name: str = FINGPT_MODEL_NAME,
    base_model_name: str = FINGPT_BASE_MODEL,
) -> ModelWrapper:
    try:
        import torch  
        from transformers import AutoTokenizer, AutoModelForCausalLM  
        from peft import PeftModel  
    except ImportError as e:
        raise ImportError(
            "Install dependencies: pip install torch transformers peft accelerate sentencepiece"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=DEVICE_MAP,
        torch_dtype=getattr(torch, "float16"),
    )
    model = PeftModel.from_pretrained(base_model, lora_model_name)
    model.eval()

    def _generate(prompt: str) -> str:
        full_prompt = (
            "[INST] You are a financial risk analyst.\n\n"
            f"{prompt} [/INST]"
        )
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3800, # Capped securely for the Llama-2 architecture 
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=FINGPT_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.001, 
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    return ModelWrapper(name=f"fingpt::{lora_model_name}", generate_fn=_generate, max_chars=FINGPT_MAX_CHARS, is_fingpt=True)


def parse_fingpt_response(raw: str) -> Dict:
    """STEP 4: Regex parser specifically for FinGPT's simple integer output."""
    out = {
        "pred_label": -1,
        "risk_score": None,
        "rationale": "FinGPT integer parse",
        "parse_ok": False,
    }
    # Look for the last number in the string (ignores intermediate hallucinated text)
    matches = re.findall(r'\d+', raw)
    if matches:
        risk = int(matches[-1])
        risk = max(0, min(100, risk))
        out["risk_score"] = risk
        out["pred_label"] = 1 if risk >= 50 else 0
        out["parse_ok"] = True
    return out


# =========================
# Evaluation pipeline
# =========================
def evaluate_model_on_split(
    model: ModelWrapper,
    df: pd.DataFrame,
    condition: str,
    prompt_name: str,
    prompt_template: str,
    fewshot_examples: List[Dict],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    
    raw_fewshot_block = format_fewshot_block(fewshot_examples, condition, model.max_chars)
    rows: List[Dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{model.name} | {prompt_name} | {condition}"):
        text = truncate_text(row.get(condition, ""), model.max_chars)
        
        if model.is_fingpt:
            # Strip the JSON instruction from the template
            current_prompt = prompt_template.split("Return ONLY valid JSON")[0]
            current_prompt += "Return ONLY a single integer from 0 to 100 representing the risk score."
            
            # Strip the JSON dict from the few-shot examples and leave only the risk_score integer
            fewshot_block = re.sub(
                r'\{"label": \d, "risk_score": (\d+), "rationale": "[^"]+"\}', 
                r'\1', 
                raw_fewshot_block
            )
        else:
            current_prompt = prompt_template
            fewshot_block = raw_fewshot_block

        prompt = current_prompt.format(few_shot_block=fewshot_block, text=text)

        try:
            raw = model.generate_fn(prompt)
        except Exception as e:
            raw = f"ERROR: {e}"

        if model.is_fingpt:
            parsed = parse_fingpt_response(raw)
        else:
            parsed = parse_model_response(raw)

        pred_label = parsed["pred_label"]
        risk_score = parsed["risk_score"]

        rows.append(
            {
                "transcript_id": row.get("transcript_id", ""),
                "ticker": row.get("ticker", ""),
                "true_label": int(row["label"]),
                "pred_label": pred_label,
                "risk_score": risk_score,
                "rationale": parsed["rationale"],
                "parse_ok": parsed["parse_ok"],
                "model": model.name,
                "prompt_name": prompt_name,
                "condition": condition,
                "raw_response": raw[:1200],
            }
        )

        if REQUEST_SLEEP_SEC > 0:
            time.sleep(REQUEST_SLEEP_SEC)

    pred_df = pd.DataFrame(rows)

    valid = pred_df[(pred_df["pred_label"].isin([0, 1])) & (pred_df["risk_score"].notna())].copy()
    parse_rate = float(pred_df["parse_ok"].mean()) if len(pred_df) else 0.0

    summary: Dict[str, float] = {
        "parse_rate": parse_rate,
        "n_total": int(len(pred_df)),
        "n_valid": int(len(valid)),
    }

    if len(valid) > 0 and valid["true_label"].nunique() > 1:
        y_true = valid["true_label"].to_numpy()
        y_score = valid["risk_score"].to_numpy() / 100.0
        y_pred = valid["pred_label"].to_numpy()
        summary.update(calculate_metrics(y_true, y_score, y_pred))
    else:
        summary.update({
            "AUROC": np.nan,
            "PR_AUC": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
        })

    return pred_df, summary


def select_best_prompt_on_validation(
    model: ModelWrapper,
    val_df: pd.DataFrame,
    condition: str,
    fewshot_examples: List[Dict],
) -> Tuple[str, pd.DataFrame]:
    records: List[Dict] = []

    for prompt_name, prompt_template in PROMPT_VARIANTS.items():
        _, summary = evaluate_model_on_split(
            model=model,
            df=val_df,
            condition=condition,
            prompt_name=prompt_name,
            prompt_template=prompt_template,
            fewshot_examples=fewshot_examples,
        )
        records.append({
            "prompt_name": prompt_name,
            **summary,
        })

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values(
        by=["AUROC", "PR_AUC", "parse_rate"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    best_prompt_name = str(results_df.iloc[0]["prompt_name"])
    return best_prompt_name, results_df


def main() -> None:
    train_df = pd.read_csv(os.path.join(DATA_DIR, "splits", "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "splits", "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "splits", "test.csv"))

    models: List[ModelWrapper] = []

    print("Loading Llama via Ollama...")
    models.append(build_ollama_llama_wrapper())

    if USE_FINGPT:
        print("Loading FinGPT locally...")
        models.append(build_fingpt_wrapper())

    all_test_predictions: List[pd.DataFrame] = []
    all_validation_prompt_results: List[pd.DataFrame] = []
    final_summaries: List[Dict] = []

    for model in models:
        for condition in CONDITIONS:
            fewshot_examples = choose_fixed_fewshot_examples(train_df, condition)

            best_prompt_name, val_prompt_df = select_best_prompt_on_validation(
                model=model,
                val_df=val_df,
                condition=condition,
                fewshot_examples=fewshot_examples,
            )
            val_prompt_df["model"] = model.name
            val_prompt_df["condition"] = condition
            all_validation_prompt_results.append(val_prompt_df)

            best_template = PROMPT_VARIANTS[best_prompt_name]
            test_pred_df, test_summary = evaluate_model_on_split(
                model=model,
                df=test_df,
                condition=condition,
                prompt_name=best_prompt_name,
                prompt_template=best_template,
                fewshot_examples=fewshot_examples,
            )
            all_test_predictions.append(test_pred_df)
            final_summaries.append(
                {
                    "model": model.name,
                    "condition": condition,
                    "selected_prompt": best_prompt_name,
                    **test_summary,
                }
            )

    validation_df = pd.concat(all_validation_prompt_results, ignore_index=True)
    validation_df.to_csv(os.path.join(OUTPUT_DIR, "tier3_validation_prompt_selection.csv"), index=False)

    test_pred_df = pd.concat(all_test_predictions, ignore_index=True)
    test_pred_df.to_csv(os.path.join(OUTPUT_DIR, "tier3_test_raw_predictions.csv"), index=False)

    final_summary_df = pd.DataFrame(final_summaries)
    final_summary_df.to_csv(os.path.join(OUTPUT_DIR, "tier3_test_summary.csv"), index=False)

    print("\nSaved files:")
    print(os.path.join(OUTPUT_DIR, "tier3_validation_prompt_selection.csv"))
    print(os.path.join(OUTPUT_DIR, "tier3_test_raw_predictions.csv"))
    print(os.path.join(OUTPUT_DIR, "tier3_test_summary.csv"))

    print("\nFinal test summary:")
    display_cols = [
        "model", "condition", "selected_prompt", "AUROC", "PR_AUC",
        "Precision", "Recall", "F1", "parse_rate", "n_valid", "n_total"
    ]
    print(final_summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()