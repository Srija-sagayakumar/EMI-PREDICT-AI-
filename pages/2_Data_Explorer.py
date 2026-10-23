import json
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import REPORT_DIR, SAMPLE_PATH, TARGET_CLF, TARGET_REG

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
st.title("📊 Data Explorer")

if not SAMPLE_PATH.exists():
    st.error("Sample data not found. Run `python -m src.train` first.")
    st.stop()


@st.cache_data
def load():
    return pd.read_csv(SAMPLE_PATH)


df = load()
scen = st.multiselect("EMI scenario", sorted(df["emi_scenario"].unique()),
                      default=sorted(df["emi_scenario"].unique()))
d = df[df["emi_scenario"].isin(scen)]
st.caption(f"{len(d):,} records")

a, b = st.columns(2)
a.plotly_chart(px.pie(d, names=TARGET_CLF, hole=.4, title="EMI eligibility distribution"),
               use_container_width=True)
b.plotly_chart(px.histogram(d, x="emi_scenario", color=TARGET_CLF, barnorm="percent",
                            title="Eligibility by EMI scenario"), use_container_width=True)
a.plotly_chart(px.box(d, x=TARGET_CLF, y="credit_score", title="Credit score vs eligibility"),
               use_container_width=True)
s = d.sample(min(3000, len(d)), random_state=1)
b.plotly_chart(px.scatter(s, x="monthly_salary", y=TARGET_REG, color=TARGET_CLF, opacity=.6,
                          title="Salary vs maximum EMI"), use_container_width=True)
st.plotly_chart(px.imshow(d.select_dtypes("number").corr(), color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, aspect="auto", height=650, title="Correlation matrix"),
                use_container_width=True)
with st.expander("Statistical summary"):
    st.dataframe(d.describe().T, use_container_width=True)
q = REPORT_DIR / "data_quality.json"
if q.exists():
    with st.expander("Data quality report"):
        st.json(json.loads(q.read_text()))
