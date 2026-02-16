import uuid
import re
import pandas as pd
from typing import List
from pathlib import Path
from django.conf import settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .models import (
    ensure_user_dir, paths, load_history, load_pickle, 
    save_pickle, load_account_map, load_semantic_map
)

ACCOUNT_REGEX = re.compile(r"\b\d{9,18}\b")
NON_ALPHA_REGEX = re.compile(r"[^a-z\s]")
MULTISPACE_REGEX = re.compile(r"\s+")
PARTICULARS_ALIASES = {
    "particulars",
    "narration",
    "description",
    "details",
    "transaction details",
    "transaction description",
    "remarks",
    "remark",
    "note",
    "memo",
}

MIN_VALIDATION_ROWS = 30
MIN_VALIDATION_BANKS = 1
LOW_SIMILARITY_FLOOR = 0.05
WARNING_SIMILARITY_THRESHOLD = 0.08
WARNING_MARGIN_THRESHOLD = 0.02
MAX_UPLOADED_DESCRIPTIONS = 500

_VALIDATION_CACHE = {
    "signature": None,
    "vectorizer": None,
    "matrix": None,
    "bankids": [],
    "is_sufficient": False,
}


class BankValidationWarning(Exception):
    def __init__(self, result):
        self.result = result
        super().__init__("Bank mismatch validation warning")

def normalize(text):
    return str(text if text is not None else "").upper().strip()

def clean_tokens(text):
    text = normalize(text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^A-Z ]", " ", text)
    return [t for t in text.split() if len(t) >= 2]


def normalize_description_for_validation(text):
    value = str(text if text is not None else "").lower()
    value = re.sub(r"\d+", " ", value)
    value = NON_ALPHA_REGEX.sub(" ", value)
    value = MULTISPACE_REGEX.sub(" ", value).strip()
    return value


def _find_matching_column(columns, candidates):
    lower_map = {str(c).strip().lower(): c for c in columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    return None


def _history_files_for_validation():
    media_root = Path(settings.MEDIA_ROOT)
    files = list(media_root.rglob("history.csv"))
    project_history = Path(settings.BASE_DIR) / "history.csv"
    if project_history.exists():
        files.append(project_history)
    return sorted(set(files))


def _history_signature(files):
    signature = []
    for file_path in files:
        stat = file_path.stat()
        signature.append((str(file_path), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def _load_history_descriptions_for_validation(files):
    rows = []
    for file_path in files:
        try:
            df = pd.read_csv(file_path)
        except Exception:
            continue

        bank_col = _find_matching_column(df.columns, ["bankid", "bank_id", "bank"])
        desc_col = _find_matching_column(
            df.columns,
            ["particulars",
    "narration",
    "description",
    "details",
    "transaction details",
    "transaction description",
    "remarks",
    "remark",
    "note",
    "memo",],
        )
        if bank_col is None or desc_col is None:
            continue

        subset = df[[bank_col, desc_col]].rename(
            columns={bank_col: "bankid", desc_col: "description"}
        )
        subset["bankid"] = subset["bankid"].astype(str).str.strip().str.lower()
        subset["description"] = subset["description"].map(normalize_description_for_validation)
        subset = subset[
            (subset["bankid"] != "")
            & (subset["description"] != "")
        ]
        if not subset.empty:
            rows.append(subset)

    if not rows:
        return pd.DataFrame(columns=["bankid", "description"])
    return pd.concat(rows, ignore_index=True)


def _ensure_validation_cache():
    files = _history_files_for_validation()
    signature = _history_signature(files)
    if _VALIDATION_CACHE["signature"] == signature:
        return

    history_df = _load_history_descriptions_for_validation(files)
    _VALIDATION_CACHE["signature"] = signature
    _VALIDATION_CACHE["vectorizer"] = None
    _VALIDATION_CACHE["matrix"] = None
    _VALIDATION_CACHE["bankids"] = []
    _VALIDATION_CACHE["is_sufficient"] = False

    if len(history_df) < MIN_VALIDATION_ROWS:
        return
    if history_df["bankid"].nunique() < MIN_VALIDATION_BANKS:
        return

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(history_df["description"])
    _VALIDATION_CACHE["vectorizer"] = vectorizer
    _VALIDATION_CACHE["matrix"] = matrix
    _VALIDATION_CACHE["bankids"] = history_df["bankid"].tolist()
    _VALIDATION_CACHE["is_sufficient"] = True


def validate_selected_bankid(descriptions, selected_bankid):
    selected_bank = str(selected_bankid).strip().lower()
    normalized_uploaded = [
        normalize_description_for_validation(text)
        for text in descriptions[:MAX_UPLOADED_DESCRIPTIONS]
    ]
    normalized_uploaded = [text for text in normalized_uploaded if text]
    if not normalized_uploaded:
        return {
            "status": "insufficient_data",
            "predicted_bank": selected_bank,
            "scores": {},
        }

    _ensure_validation_cache()
    if not _VALIDATION_CACHE["is_sufficient"]:
        return {
            "status": "insufficient_data",
            "predicted_bank": selected_bank,
            "scores": {},
        }

    vectorizer = _VALIDATION_CACHE["vectorizer"]
    history_matrix = _VALIDATION_CACHE["matrix"]
    history_bankids = _VALIDATION_CACHE["bankids"]

    uploaded_matrix = vectorizer.transform(normalized_uploaded)
    similarity_matrix = cosine_similarity(uploaded_matrix, history_matrix)

    unique_banks = sorted(set(history_bankids))
    scores = {}
    for bank in unique_banks:
        indices = [idx for idx, bankid in enumerate(history_bankids) if bankid == bank]
        if not indices:
            continue
        # For each uploaded row, take its best match within this bank, then average.
        # This avoids dilution from many unrelated historical rows.
        bank_block = similarity_matrix[:, indices]
        bank_score = float(bank_block.max(axis=1).mean())
        scores[bank] = bank_score

    if not scores:
        return {
            "status": "insufficient_data",
            "predicted_bank": selected_bank,
            "scores": {},
        }

    predicted_bank = max(scores, key=scores.get)
    predicted_score = scores[predicted_bank]
    selected_score = scores.get(selected_bank, 0.0)

    status = "ok"
    if predicted_score >= LOW_SIMILARITY_FLOOR:
        if (
            predicted_bank != selected_bank
            and predicted_score >= WARNING_SIMILARITY_THRESHOLD
            and (predicted_score - selected_score) >= WARNING_MARGIN_THRESHOLD
        ):
            status = "warning"

    return {
        "status": status,
        "predicted_bank": predicted_bank,
        "scores": {bank: round(score, 6) for bank, score in scores.items()},
    }

def longest_common_subsequence(a: List[str], b: List[str]) -> List[str]:
    dp = [[[] for _ in range(len(b) + 1)] for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + [a[i]]
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j], key=len)
    return dp[-1][-1]

def promote_semantic_anchor(user_id, bankid, acname):
    history = load_history(user_id, bankid)
    rows = history[history["acname"] == acname]

    rows = rows[~rows["Particulars"].astype(str).str.contains(ACCOUNT_REGEX)]
    if len(rows) < 2:
        return

    tokens_list = [clean_tokens(t) for t in rows["Particulars"]]
    common = tokens_list[0]
    for tokens in tokens_list[1:]:
        common = longest_common_subsequence(common, tokens)
        if not common:
            return

    if len(common) >= 3:
        semantic_map = load_pickle(paths(user_id, bankid)["semantic"])
        semantic_map[" ".join(common)] = acname
        save_pickle(paths(user_id, bankid)["semantic"], semantic_map)

def detect_particulars_column(df):
   
    for col in df.columns:
        key = str(col).strip().lower()
        if key in PARTICULARS_ALIASES:
            return col

    best_col, best_score = None, -1
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        non_empty = series.astype(str).str.strip() != ""
        if not non_empty.any():
            continue
        sample = series[non_empty].astype(str).head(200)
        token_counts = sample.apply(lambda s: len(clean_tokens(s)))
        avg_tokens = token_counts.mean()
        if avg_tokens > best_score:
            best_score = avg_tokens
            best_col = col
    return best_col

def run_prediction_file(file_path, user_id, bankid, enforce_bank_validation=True):
    df = pd.read_excel(file_path) if file_path.endswith(".xlsx") else pd.read_csv(file_path)
    df.columns = [str(c).strip() for c in df.columns]

    target = detect_particulars_column(df)
    if not target:
        raise ValueError(
            "Missing particulars column. Please rename the column to 'Particulars'. "
            f"Found: {list(df.columns)}"
        )
    
    df.rename(columns={target: "Particulars"}, inplace=True)
    df["Particulars"] = df["Particulars"].fillna("").astype(str)

    if (df["Particulars"].str.strip() == "").any():
        raise ValueError("All rows must have a non-empty Particulars value")

    if enforce_bank_validation:
        validation = validate_selected_bankid(df["Particulars"].tolist(), bankid)
        if validation["status"] == "warning":
            warning_payload = {
                **validation,
                "message": (
                    f"Selected bank '{bankid}' does not match statement pattern. "
                    f"Predicted bank: '{validation['predicted_bank']}'."
                ),
                "error": (
                    f"Selected bank '{bankid}' may be incorrect. "
                    f"Predicted bank: '{validation['predicted_bank']}'."
                ),
            }
            raise BankValidationWarning(warning_payload)

    labels, sources = [], []
    for t in df["Particulars"]:
        l, s = predict_acname(t, user_id, bankid)
        labels.append(l)
        sources.append(s)

    df["acname"], df["source"] = labels, sources
    df["OrgID"], df["BankID"] = user_id, bankid
    return df

def predict_acname(particulars, user_id, bankid):
    ensure_user_dir(user_id, bankid)
    history = load_history(user_id, bankid)
    account_map = load_account_map(user_id, bankid)
    semantic_map = load_semantic_map(user_id, bankid)

    acc_match = ACCOUNT_REGEX.search(particulars)
    if acc_match and acc_match.group(0) in account_map:
        return account_map[acc_match.group(0)], "ACCOUNT"

    tokens = clean_tokens(particulars)
    for anchor, label in semantic_map.items():
        anchor_tokens = anchor.split()
        if all(tok in tokens for tok in anchor_tokens):
            return label, "SEMANTIC"

    best_score, best_label = 0.0, ""
    a = set(tokens)
    for _, row in history.iterrows():
        b = set(clean_tokens(row["Particulars"]))
        union = a | b
        if union:
            score = len(a & b) / len(union)
            if score > best_score:
                best_score, best_label = score, row["acname"]

    if best_score >= 0.6:
        return best_label, "MEMORY"
    return "", "UNKNOWN"

def append_history(user_id, bankid, particulars, acname):
    ensure_user_dir(user_id, bankid)
    df = load_history(user_id, bankid)
    particulars = str(particulars if particulars is not None else "").strip()
    acname = str(acname if acname is not None else "")

    if particulars:
        mask = df["Particulars"].astype(str).str.strip() == particulars
        if mask.any():
            df.loc[mask, "acname"] = acname
        else:
            new_row = pd.DataFrame([{
                "ID": str(uuid.uuid4()),
                "OrgID": user_id,
                "BankID": bankid,
                "Particulars": particulars,
                "acname": acname
            }])
            df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(paths(user_id, bankid)["history"], index=False)

def learn_account_mapping(user_id, bankid, particulars, acname):
    acc_match = ACCOUNT_REGEX.search(particulars)
    if acc_match:
        p = paths(user_id, bankid)["account"]
        m = load_pickle(p)
        m[acc_match.group(0)] = acname
        save_pickle(p, m)
