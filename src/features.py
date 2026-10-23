"""Feature engineering + preprocessing pipeline (shared by training and the app)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import CAT_RAW, NUM_RAW

ENGINEERED = [
    "total_expenses", "expense_to_income", "debt_to_income", "disposable_income",
    "requested_emi", "emi_to_income", "post_emi_dti", "affordability_ratio",
    "loan_to_income", "savings_cushion_months", "liquid_assets_ratio",
    "credit_risk_score", "employment_stability", "dependents_ratio", "salary_x_credit",
]
_EXPENSE_COLS = ["monthly_rent", "school_fees", "college_fees", "travel_expenses",
                 "groceries_utilities", "other_monthly_expenses"]
_EMP_WEIGHT = {"Government": 1.2, "Private": 1.0, "Self-employed": 0.8}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived financial ratios, risk scores and interaction features."""
    d = df.copy()
    sal = d["monthly_salary"].clip(lower=1)
    d["total_expenses"] = d[_EXPENSE_COLS].sum(axis=1)
    d["expense_to_income"] = (d["total_expenses"] / sal).clip(0, 5)
    d["debt_to_income"] = (d["current_emi_amount"] / sal).clip(0, 5)
    d["disposable_income"] = d["monthly_salary"] - d["total_expenses"] - d["current_emi_amount"]
    d["requested_emi"] = d["requested_amount"] / d["requested_tenure"].clip(lower=1)
    d["emi_to_income"] = (d["requested_emi"] / sal).clip(0, 5)
    d["post_emi_dti"] = ((d["current_emi_amount"] + d["requested_emi"]) / sal).clip(0, 5)
    d["affordability_ratio"] = (d["requested_emi"] / d["disposable_income"].clip(lower=1)).clip(0, 20)
    d["loan_to_income"] = (d["requested_amount"] / (sal * 12)).clip(0, 20)
    burn = (d["total_expenses"] + d["current_emi_amount"]).clip(lower=1)
    assets = d["bank_balance"] + d["emergency_fund"]
    d["savings_cushion_months"] = (assets / burn).clip(0, 120)
    d["liquid_assets_ratio"] = (assets / (sal * 12)).clip(0, 20)
    d["credit_risk_score"] = ((d["credit_score"] - 300) / 550).clip(0, 1)
    w = d["employment_type"].map(_EMP_WEIGHT).astype(float).fillna(1.0)
    d["employment_stability"] = np.log1p(d["years_of_employment"].clip(lower=0)) * w
    d["dependents_ratio"] = d["dependents"] / d["family_size"].clip(lower=1)
    d["salary_x_credit"] = sal * d["credit_score"] / 1000
    return d.replace([np.inf, -np.inf], np.nan)


def build_preprocessor() -> Pipeline:
    ct = ColumnTransformer(
        [("num", StandardScaler(), NUM_RAW + ENGINEERED),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_RAW)]
    )
    return Pipeline([("fe", FunctionTransformer(add_features)), ("ct", ct)])
