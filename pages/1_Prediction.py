import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_DIR
from src.utils import category_options

st.set_page_config(page_title="EMI Prediction", page_icon="🔮", layout="wide")
st.title("🔮 EMI Prediction")


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    return (joblib.load(MODEL_DIR / "classifier.joblib"),
            joblib.load(MODEL_DIR / "regressor.joblib"))


try:
    clf, reg = load_models()
except FileNotFoundError:
    st.error("Models not found. Run `python -m src.train` first.")
    st.stop()

cats = category_options()
opt = lambda col, default: cats.get(col) or default

with st.form("applicant"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Personal & employment")
        age = st.number_input("Age", 18, 70, 35)
        gender = st.selectbox("Gender", opt("gender", ["Male", "Female"]))
        marital_status = st.selectbox("Marital status", opt("marital_status", ["Single", "Married"]))
        education = st.selectbox("Education", opt("education", ["High School", "Graduate", "Post Graduate", "Professional"]))
        employment_type = st.selectbox("Employment type", opt("employment_type", ["Private", "Government", "Self-employed"]))
        company_type = st.selectbox("Company type", opt("company_type", ["Small", "Mid-size", "Large Indian", "MNC", "Startup"]))
        years_of_employment = st.number_input("Years of employment", 0.0, 45.0, 8.0, 0.5)
        monthly_salary = st.number_input("Monthly salary (₹)", 0.0, 500000.0, 60000.0, 1000.0)
    with c2:
        st.subheader("Household & expenses")
        house_type = st.selectbox("House type", opt("house_type", ["Rented", "Own", "Family"]))
        monthly_rent = st.number_input("Monthly rent (₹)", 0.0, 200000.0, 12000.0, 500.0)
        family_size = st.number_input("Family size", 1, 15, 4)
        dependents = st.number_input("Dependents", 0, 14, 2)
        school_fees = st.number_input("School fees (₹)", 0.0, 100000.0, 3000.0, 500.0)
        college_fees = st.number_input("College fees (₹)", 0.0, 200000.0, 0.0, 500.0)
        travel_expenses = st.number_input("Travel expenses (₹)", 0.0, 100000.0, 3000.0, 500.0)
        groceries_utilities = st.number_input("Groceries & utilities (₹)", 0.0, 200000.0, 10000.0, 500.0)
        other_monthly_expenses = st.number_input("Other monthly expenses (₹)", 0.0, 200000.0, 3000.0, 500.0)
    with c3:
        st.subheader("Credit & loan request")
        existing_loans = st.selectbox("Existing loans", opt("existing_loans", ["No", "Yes"]))
        current_emi_amount = st.number_input("Current EMI burden (₹)", 0.0, 300000.0, 0.0, 500.0)
        credit_score = st.slider("Credit score", 300, 850, 700)
        bank_balance = st.number_input("Bank balance (₹)", 0.0, 10_000_000.0, 150000.0, 5000.0)
        emergency_fund = st.number_input("Emergency fund (₹)", 0.0, 10_000_000.0, 60000.0, 5000.0)
        emi_scenario = st.selectbox("EMI scenario", opt("emi_scenario", ["Personal Loan EMI"]))
        requested_amount = st.number_input("Requested amount (₹)", 1000.0, 2_000_000.0, 300000.0, 5000.0)
        requested_tenure = st.number_input("Requested tenure (months)", 3, 84, 24)
    go = st.form_submit_button("Assess applicant", type="primary", use_container_width=True)

if go:
    row = dict(age=age, gender=gender, marital_status=marital_status, education=education,
               monthly_salary=monthly_salary, employment_type=employment_type,
               years_of_employment=years_of_employment, company_type=company_type,
               house_type=house_type, monthly_rent=monthly_rent, family_size=family_size,
               dependents=dependents, school_fees=school_fees, college_fees=college_fees,
               travel_expenses=travel_expenses, groceries_utilities=groceries_utilities,
               other_monthly_expenses=other_monthly_expenses, existing_loans=existing_loans,
               current_emi_amount=current_emi_amount, credit_score=credit_score,
               bank_balance=bank_balance, emergency_fund=emergency_fund,
               emi_scenario=emi_scenario, requested_amount=requested_amount,
               requested_tenure=requested_tenure)
    X = pd.DataFrame([row])
    try:
        proba = clf["pipeline"].predict_proba(X)[0]
        label = clf["classes"][int(proba.argmax())]
        max_emi = max(float(reg["pipeline"].predict(X)[0]), 0.0)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.divider()
    r1, r2 = st.columns(2)
    r1.metric("EMI eligibility", label.replace("_", " "), f"{proba.max()*100:.1f}% confidence", delta_color="off")
    r2.metric("Maximum monthly EMI", f"₹{max_emi:,.0f}")
    st.bar_chart(pd.Series(proba, index=[c.replace("_", " ") for c in clf["classes"]]))
