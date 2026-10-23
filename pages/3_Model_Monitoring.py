import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import MLFLOW_DB, MLFLOW_URI, MODEL_DIR
from src.utils import load_meta

st.set_page_config(page_title="Model Monitoring", page_icon="📈", layout="wide")
st.title("📈 Model Performance & MLflow")
meta = load_meta()
if not meta:
    st.error("No results yet. Run `python -m src.train` first.")
    st.stop()

st.success(f"Deployed → classifier: **{meta['best_classifier']}** · regressor: **{meta['best_regressor']}**")

t1, t2, t3 = st.tabs(["Classification", "Regression", "MLflow runs"])
with t1:
    df = pd.read_csv(MODEL_DIR / "clf_metrics.csv")
    cols = ["model", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    st.dataframe(df[cols].round(4), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(df.melt(id_vars="model", value_vars=cols[1:]), x="variable", y="value",
                           color="model", barmode="group"), use_container_width=True)
with t2:
    df = pd.read_csv(MODEL_DIR / "reg_metrics.csv")
    cols = ["model", "test_rmse", "test_mae", "test_r2", "test_mape"]
    st.dataframe(df[cols].round(4), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(df, x="model", y="test_rmse", color="model", title="Test RMSE"),
                    use_container_width=True)
with t3:
    if MLFLOW_DB.exists():
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            runs = mlflow.search_runs(experiment_names=["EMIPredict_AI"])
            keep = [c for c in runs.columns if c.startswith(("tags.task", "tags.model_name", "metrics.val_"))]
            st.dataframe(runs[keep], use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not read MLflow store: {e}")
    else:
        st.info("MLflow database not available on this server.")
    st.code(f"mlflow ui --backend-store-uri {MLFLOW_URI}", language="bash")
