"""Central configuration: paths, column groups, MLflow settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "EMI_dataset.csv"        # <- put the downloaded dataset here
SAMPLE_PATH = DATA_DIR / "emi_sample.csv"      # cleaned sample used by the web app
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
PLOT_DIR = REPORT_DIR / "plots"
ARTIFACT_DIR = ROOT / "mlartifacts"
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

for _d in (DATA_DIR, MODEL_DIR, REPORT_DIR, PLOT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_CLF = "emi_eligibility"
TARGET_REG = "max_monthly_emi"
TARGETS = [TARGET_CLF, TARGET_REG]

NUM_RAW = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent", "family_size",
    "dependents", "school_fees", "college_fees", "travel_expenses",
    "groceries_utilities", "other_monthly_expenses", "current_emi_amount",
    "credit_score", "bank_balance", "emergency_fund", "requested_amount",
    "requested_tenure",
]
CAT_RAW = [
    "gender", "marital_status", "education", "employment_type", "company_type",
    "house_type", "existing_loans", "emi_scenario",
]
ALL_FEATURES = NUM_RAW + CAT_RAW
