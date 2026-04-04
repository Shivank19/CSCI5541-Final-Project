import re
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

# FEATURE EXTRACTION PIPELINE
class LoughranMcDonaldExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, dict_path: str):
        self.dict_path = dict_path
        self._categories = [
            "Negative",
            "Positive",
            "Uncertainty",
            "Litigious",
            "Strong_Modal",
            "Weak_Modal",
        ]
        self._lexicons = self._load_lexicons()

    def _load_lexicons(self) -> dict:
        lexicons = {cat: set() for cat in self._categories}
        try:
            df = pd.read_csv(self.dict_path)
            df = df.dropna(subset=["Word"])
            for cat in self._categories:
                words = df[df[cat] > 0]["Word"].str.lower().tolist()
                lexicons[cat] = set(words)
        except Exception as e:
            print(f"Error loading dictionary: {e}")
        return lexicons

    def _tokenize(self, text: str) -> list:
        if not isinstance(text, str):
            return []
        return re.findall(r"\b[a-z]+\b", text.lower())

    def _get_term_frequencies(self, text: str) -> list:
        tokens = self._tokenize(text)
        total_words = len(tokens)
        if total_words == 0:
            return [0.0] * len(self._categories)

        freqs = []
        for cat in self._categories:
            count = sum(1 for word in tokens if word in self._lexicons[cat])
            freqs.append(count / total_words)
        return freqs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        features = [self._get_term_frequencies(text) for text in X]
        return np.array(features)

    def get_feature_names_out(self, input_features=None):
        # Prefixed with LM_ to distinguish from TF-IDF unigrams
        return np.array([f"LM_{cat}" for cat in self._categories])


def build_feature_pipeline(lm_dict_path: str) -> FeatureUnion:
    pipeline = FeatureUnion(
        [
            ("lm_lexicon", LoughranMcDonaldExtractor(dict_path=lm_dict_path)),
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english", ngram_range=(1, 1), max_df=0.95, min_df=2
                ),
            ),
        ]
    )
    return pipeline


# EVALUATION METRICS
def calculate_metrics(y_true, y_prob):
    metrics = {}
    metrics["AUROC"] = roc_auc_score(y_true, y_prob)
    metrics["PR_AUC"] = average_precision_score(y_true, y_prob)

    y_pred_hr = (y_prob >= 0.3).astype(int)
    metrics["HR_Prec"] = precision_score(y_true, y_pred_hr, zero_division=0)
    metrics["HR_Rec"] = recall_score(y_true, y_pred_hr, zero_division=0)
    metrics["HR_F1"] = f1_score(y_true, y_pred_hr, zero_division=0)

    y_pred_hp = (y_prob >= 0.7).astype(int)
    metrics["HP_Prec"] = precision_score(y_true, y_pred_hp, zero_division=0)
    metrics["HP_Rec"] = recall_score(y_true, y_pred_hp, zero_division=0)
    metrics["HP_F1"] = f1_score(y_true, y_pred_hp, zero_division=0)

    return metrics


# SHAP EXTRACTOR
def extract_shap_importances(
    model, X_test_feats, feature_names, condition_name, top_n=15
):
    """
    Extracts and prints the top driving features using SHAP TreeExplainer.
    """
    print(
        f"\n--- Top {top_n} SHAP Features for {condition_name.upper()} (Random Forest) ---"
    )

    # Convert sparse matrix or object array to a standard dense float array for SHAP
    if hasattr(X_test_feats, "toarray"):
        X_test_feats = X_test_feats.toarray()
    X_test_feats = np.asarray(X_test_feats, dtype=float)

    # We use TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_feats)

    # Handle different SHAP output formats depending on version
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        # Some newer versions of shap return a 3D array for RF
        shap_values_pos = (
            shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
        )

    # Calculate mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Mean_Abs_SHAP": mean_abs_shap}
    )
    importance_df = importance_df.sort_values(by="Mean_Abs_SHAP", ascending=False).head(
        top_n
    )

    print(importance_df.to_string(index=False))
    return importance_df


# MAIN EXECUTION LOOP
def run_baseline(
    data_dir: str, lm_dict_path: str, output_csv: str = "tier1_baseline_results.csv"
):
    train_df = pd.read_csv(f"{data_dir}/splits/train.csv")
    val_df = pd.read_csv(f"{data_dir}/splits/val.csv")
    test_df = pd.read_csv(f"{data_dir}/splits/test.csv")

    y_train = train_df["label"]
    y_val = val_df["label"]
    y_test = test_df["label"]

    conditions = ["full_text", "scripted_text", "qa_text"]

    all_results = []
    saved_rf_models = {}
    saved_test_feats = {}
    saved_feature_names = {}

    for condition in conditions:
        print(f"\n--- Processing Condition: {condition.upper()} ---")

        X_train = train_df[condition].fillna("")
        X_val = val_df[condition].fillna("")
        X_test = test_df[condition].fillna("")

        pipeline = build_feature_pipeline(lm_dict_path)
        print("Extracting features...")
        X_train_feats = pipeline.fit_transform(X_train)
        X_val_feats = pipeline.transform(X_val)
        X_test_feats = pipeline.transform(X_test)

        # Extract feature names to map back SHAP values later
        lm_names = pipeline.named_transformers["lm_lexicon"].get_feature_names_out()
        tfidf_names = pipeline.named_transformers["tfidf"].get_feature_names_out()

        feature_names = np.concatenate(
            [
                [f"lm_lexicon__{name}" for name in lm_names],
                [f"tfidf__{name}" for name in tfidf_names],
            ]
        )

        saved_test_feats[condition] = X_test_feats
        saved_feature_names[condition] = feature_names

        classifiers = {
            "Logistic Regression": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            ),
            "SVM (Linear)": SVC(
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=42,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42
            ),
        }

        for clf_name, clf in classifiers.items():
            clf.fit(X_train_feats, y_train)

            # Save the Random Forest model for SHAP analysis
            if clf_name == "Random Forest":
                saved_rf_models[condition] = clf

            train_probs = clf.predict_proba(X_train_feats)[:, 1]
            val_probs = clf.predict_proba(X_val_feats)[:, 1]
            test_probs = clf.predict_proba(X_test_feats)[:, 1]

            train_metrics = calculate_metrics(y_train, train_probs)
            val_metrics = calculate_metrics(y_val, val_probs)
            test_metrics = calculate_metrics(y_test, test_probs)

            for split_name, metrics in zip(
                ["Train", "Validation", "Test"],
                [train_metrics, val_metrics, test_metrics],
            ):
                row = {"Condition": condition, "Model": clf_name, "Split": split_name}
                row.update(metrics)
                all_results.append(row)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nEvaluation complete. Full metrics saved to {output_csv}")

    print("\n--- TEST SET AUROC SUMMARY ---")
    test_summary = results_df[results_df["Split"] == "Test"][
        ["Condition", "Model", "AUROC"]
    ]
    print(test_summary.to_string(index=False))

    # Run SHAP Interpretability on the Random Forest models
    print("\n--- SHAP INTERPRETABILITY ANALYSIS ---")
    for condition in ["scripted_text", "qa_text"]:
        extract_shap_importances(
            model=saved_rf_models[condition],
            X_test_feats=saved_test_feats[condition],
            feature_names=saved_feature_names[condition],
            condition_name=condition,
        )


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent

    DATA_DIR = SCRIPT_DIR.parent.parent / "data"
    LM_DICT_PATH = SCRIPT_DIR / "Loughran-McDonald_MasterDictionary_1993-2025.csv"

    run_baseline(data_dir=DATA_DIR, lm_dict_path=LM_DICT_PATH)
