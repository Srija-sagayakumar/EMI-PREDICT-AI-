# EMIPredict AI – Intelligent Financial Risk Assessment Platform

## Step-by-step (VS Code)
1. File → Open Folder → `emipredict-ai`.
2. Terminal (Ctrl+`):
   `python -m venv venv` → activate (`venv\Scripts\activate` on Windows, `source venv/bin/activate` on macOS/Linux)
   → `pip install -r requirements.txt`. Select the `venv` interpreter (Ctrl+Shift+P → Python: Select Interpreter).
3. Save the dataset as `data/EMI_dataset.csv`.
4. Preprocess: `python -m src.preprocess` (cleans data, writes `data/emi_clean.csv` + `reports/data_quality.json`).
5. EDA (optional): `python -m src.eda` (charts in `reports/eda/`).
6. Train + MLflow tracking + registry: `python -m src.train` (`--sample 100000` for a quick run).
7. MLflow dashboard: `mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://127.0.0.1:5000
8. App: `streamlit run Home.py` → http://localhost:8501

## Deploy (Streamlit Cloud)
Push to GitHub (commit `models/`, `data/emi_sample.csv`, `reports/`), then share.streamlit.io → New app → main file `Home.py`.
