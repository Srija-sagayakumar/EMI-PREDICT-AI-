import streamlit as st
from src.utils import load_meta

st.set_page_config(page_title="EMIPredict AI", page_icon="💳", layout="wide")
st.title("💳 EMIPredict AI")
st.caption("Intelligent Financial Risk Assessment Platform")
st.write("EMI eligibility (classification) and maximum monthly EMI (regression) "
         "across five EMI scenarios.")

meta = load_meta()
if meta:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best classifier", meta["best_classifier"].replace("_", " "))
    c2.metric("Accuracy (test)", f"{meta['best_clf_metrics']['test_accuracy']*100:.2f}%")
    c3.metric("Best regressor", meta["best_regressor"].replace("_", " "))
    c4.metric("RMSE (test)", f"₹{meta['best_reg_metrics']['test_rmse']:,.0f}")
else:
    st.warning("No trained models found. Run `python -m src.train` first.")

st.markdown("""
**Pages (sidebar):** Prediction · Data Explorer · Model Monitoring (MLflow) · Admin (CRUD)
""")
