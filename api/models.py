import os
import pickle
import pandas as pd
from django.conf import settings

def user_bank_base_dir(user_id, bankid):
    """Creates a base path isolated by both user and bank."""
    return os.path.join(settings.MEDIA_ROOT, "users", str(user_id), str(bankid))

def paths(user_id, bankid):
    """Mapping for all file paths used by the engine."""
    base = user_bank_base_dir(user_id, bankid)
    return {
        "base": base,
        "uploads": os.path.join(base, "uploads"),
        "history": os.path.join(base, "history.csv"),
        "account": os.path.join(base, "account_map.pkl"),
        "semantic": os.path.join(base, "semantic_map.pkl"),
    }

def ensure_user_dir(user_id, bankid):
    """Ensures the directory hierarchy exists."""
    os.makedirs(paths(user_id, bankid)["uploads"], exist_ok=True)

def load_pickle(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_history(user_id, bankid):
    p = paths(user_id, bankid)["history"]
    cols = ["ID", "OrgID", "BankID", "Particulars", "acname"]
    if os.path.exists(p):
        df = pd.read_csv(p)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        return df[cols]
    return pd.DataFrame(columns=cols)

def load_account_map(user_id, bankid):
    return load_pickle(paths(user_id, bankid)["account"])

def load_semantic_map(user_id, bankid):
    return load_pickle(paths(user_id, bankid)["semantic"])