"""Data loading, cleaning, quality assessment and train/val/test split."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (ALL_FEATURES, CAT_RAW, DATA_DIR, NUM_RAW, RANDOM_STATE, RAW_PATH,
                     REPORT_DIR, TARGET_CLF, TARGET_REG, TARGETS)


def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def clean(df: pd.DataFrame):
    """Returns (clean_df, quality_report)."""
    rep = {"rows_raw": int(len(df))}
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    absent = [c for c in ALL_FEATURES + TARGETS if c not in df.columns]
    if absent:
        raise ValueError(f"Dataset is missing expected columns: {absent}\n"
                         f"Found columns: {list(df.columns)}")
    df = df[ALL_FEATURES + TARGETS].copy()

    # --- text normalisation
    for c in CAT_RAW + [TARGET_CLF]:
        s = df[c].astype("object")
        s = s.where(s.isna(), s.astype(str).str.strip())
        df[c] = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})
    df["gender"] = df["gender"].str.title().replace({"M": "Male", "F": "Female"})
    df[TARGET_CLF] = df[TARGET_CLF].str.replace(r"[\s\-]+", "_", regex=True)

    # --- numeric coercion (strings like "45,000" or "abc" -> NaN)
    for c in NUM_RAW + [TARGET_REG]:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.replace(",", "", regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")

    rep["missing_before"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}
    rep["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    before = len(df)
    df = df.dropna(subset=TARGETS)
    rep["rows_dropped_missing_target"] = int(before - len(df))

    # --- imputation
    for c in NUM_RAW:
        df[c] = df[c].fillna(df[c].median())
    for c in CAT_RAW:
        df[c] = df[c].fillna(df[c].mode().iloc[0])

    # --- domain validation (clip out-of-range values)
    ranges = {"age": (18, 70), "credit_score": (300, 850), "family_size": (1, None),
              "requested_tenure": (1, None)}
    for c in NUM_RAW:
        ranges.setdefault(c, (0, None))
    clipped = {}
    for c, (lo, hi) in ranges.items():
        new = df[c].clip(lower=lo, upper=hi)
        n = int((new != df[c]).sum())
        if n:
            clipped[c] = n
        df[c] = new
    rep["values_clipped"] = clipped
    df[TARGET_REG] = df[TARGET_REG].clip(lower=0)

    rep["rows_clean"] = int(len(df))
    rep["class_distribution"] = {k: int(v) for k, v in df[TARGET_CLF].value_counts().items()}
    rep["target_reg_summary"] = {k: float(v) for k, v in df[TARGET_REG].describe().items()}
    return df.reset_index(drop=True), rep


def split_data(df: pd.DataFrame, val=0.15, test=0.15):
    """70 / 15 / 15 stratified train-validation-test split."""
    tr, tmp = train_test_split(df, test_size=val + test, stratify=df[TARGET_CLF],
                               random_state=RANDOM_STATE)
    va, te = train_test_split(tmp, test_size=test / (val + test), stratify=tmp[TARGET_CLF],
                              random_state=RANDOM_STATE)
    return tr, va, te


if __name__ == "__main__":
    # python -m src.preprocess  -> cleaned CSV + data-quality report
    df, rep = clean(load_raw(RAW_PATH))
    df.to_csv(DATA_DIR / "emi_clean.csv", index=False)
    (REPORT_DIR / "data_quality.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    tr, va, te = split_data(df)
    print(f"\nSplit -> train {len(tr):,} | validation {len(va):,} | test {len(te):,}")
    print("Saved data/emi_clean.csv and reports/data_quality.json")
