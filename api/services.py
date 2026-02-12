import uuid
import re
import pandas as pd
from typing import List
from .models import (
    ensure_user_dir, paths, load_history, load_pickle, 
    save_pickle, load_account_map, load_semantic_map
)

ACCOUNT_REGEX = re.compile(r"\b\d{9,18}\b")
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

def normalize(text):
    return str(text if text is not None else "").upper().strip()

def clean_tokens(text):
    text = normalize(text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^A-Z ]", " ", text)
    return [t for t in text.split() if len(t) >= 2]

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

def run_prediction_file(file_path, user_id, bankid):
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
