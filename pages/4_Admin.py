import pandas as pd
import streamlit as st

from src.config import SAMPLE_PATH, TARGET_CLF
from src.utils import category_options, load_meta

st.set_page_config(page_title="Admin", page_icon="🛠️", layout="wide")
st.title("🛠️ Admin: Data Management")

if not SAMPLE_PATH.exists():
    st.error("Run `python -m src.train` first to create the working dataset.")
    st.stop()
if "df" not in st.session_state:
    st.session_state.df = pd.read_csv(SAMPLE_PATH)
df = st.session_state.df
cats = category_options()
classes = load_meta().get("classes", ["Eligible", "High_Risk", "Not_Eligible"])


def save():
    st.session_state.df.to_csv(SAMPLE_PATH, index=False)
    st.cache_data.clear()


def record_form(prefix, defaults):
    out, cols = {}, st.columns(4)
    for i, c in enumerate(df.columns):
        with cols[i % 4]:
            key, dv = f"{prefix}_{c}", defaults.get(c)
            opts = classes if c == TARGET_CLF else cats.get(c)
            if opts:
                out[c] = st.selectbox(c, opts, index=opts.index(dv) if dv in opts else 0, key=key)
            else:
                out[c] = st.number_input(c, value=float(dv if dv is not None else df[c].median()), key=key)
    return out


st.caption(f"{len(df):,} records")
t_read, t_create, t_update, t_delete = st.tabs(["Read", "Create", "Update", "Delete"])

with t_read:
    q = st.text_input("Filter (e.g. `credit_score > 750`)")
    view = df
    if q:
        try:
            view = df.query(q)
        except Exception as e:
            st.warning(f"Invalid query: {e}")
    st.dataframe(view.head(500), use_container_width=True)

with t_create:
    with st.form("create"):
        new = record_form("new", df.iloc[0].to_dict())
        if st.form_submit_button("Add record"):
            st.session_state.df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
            save(); st.success("Record added."); st.rerun()

with t_update:
    idx = int(st.number_input("Row index", 0, len(df) - 1, 0))
    with st.form("update"):
        upd = record_form(f"upd{idx}", df.iloc[idx].to_dict())
        if st.form_submit_button("Save changes"):
            for k, v in upd.items():
                st.session_state.df.at[idx, k] = v
            save(); st.success(f"Row {idx} updated.")

with t_delete:
    rows = st.multiselect("Row indices to delete", df.index.tolist())
    if st.button("Delete selected", disabled=not rows):
        st.session_state.df = df.drop(index=rows).reset_index(drop=True)
        save(); st.success(f"Deleted {len(rows)} record(s)."); st.rerun()
