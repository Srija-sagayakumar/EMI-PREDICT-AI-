"""Train classification + regression models, track everything in MLflow, register the best.

Usage:
    python -m src.train                       # full dataset in data/EMI_dataset.csv
    python -m src.train --sample 100000       # faster run on a subset
    python -m src.train --data path/to.csv
"""
import argparse
import json
import time
import warnings
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from .config import (ALL_FEATURES, ARTIFACT_DIR, CAT_RAW, DATA_DIR, MLFLOW_URI,
                     MODEL_DIR, PLOT_DIR, RANDOM_STATE, RAW_PATH, REPORT_DIR,
                     SAMPLE_PATH, TARGET_CLF, TARGET_REG)
from .features import build_preprocessor
from .preprocess import clean, load_raw, split_data

warnings.filterwarnings("ignore")
CLF_REGISTRY_NAME = "EMI_Eligibility_Classifier"
REG_REGISTRY_NAME = "EMI_Max_Amount_Regressor"


# ----------------------------------------------------------------- models
def classifiers():
    return {
        "Logistic_Regression": LogisticRegression(max_iter=500),
        "Decision_Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=10,
                                                random_state=RANDOM_STATE),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=14,
                                                min_samples_leaf=5, n_jobs=-1,
                                                random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                 n_jobs=-1, eval_metric="mlogloss", random_state=RANDOM_STATE),
    }


def regressors():
    return {
        "Linear_Regression": LinearRegression(),
        "Decision_Tree": DecisionTreeRegressor(max_depth=14, min_samples_leaf=10,
                                               random_state=RANDOM_STATE),
        "Random_Forest": RandomForestRegressor(n_estimators=100, max_depth=14,
                                               min_samples_leaf=5, n_jobs=-1,
                                               random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=400, max_depth=8, learning_rate=0.08,
                                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                n_jobs=-1, random_state=RANDOM_STATE),
    }


# ---------------------------------------------------------------- metrics
def clf_metrics(pipe, X, y, p):
    pred, proba = pipe.predict(X), pipe.predict_proba(X)
    return {
        f"{p}_accuracy": accuracy_score(y, pred),
        f"{p}_precision": precision_score(y, pred, average="weighted", zero_division=0),
        f"{p}_recall": recall_score(y, pred, average="weighted", zero_division=0),
        f"{p}_f1": f1_score(y, pred, average="weighted", zero_division=0),
        f"{p}_roc_auc": roc_auc_score(y, proba, multi_class="ovr", average="weighted"),
    }


def reg_metrics(pipe, X, y, p):
    pred = pipe.predict(X)
    return {
        f"{p}_rmse": float(np.sqrt(mean_squared_error(y, pred))),
        f"{p}_mae": mean_absolute_error(y, pred),
        f"{p}_r2": r2_score(y, pred),
        f"{p}_mape": mean_absolute_percentage_error(np.maximum(y, 1), pred) * 100,
    }


# ------------------------------------------------------------------ plots
def plot_confusion(pipe, X, y, classes, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, pipe.predict(X), display_labels=classes,
                                            ax=ax, cmap="Blues", colorbar=False)
    plt.xticks(rotation=20); plt.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_pred_vs_actual(pipe, X, y, path):
    idx = np.random.RandomState(0).choice(len(X), min(3000, len(X)), replace=False)
    pred = pipe.predict(X.iloc[idx]); act = np.asarray(y)[idx]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(act, pred, s=5, alpha=.4)
    lim = [0, max(act.max(), pred.max())]
    ax.plot(lim, lim, "r--"); ax.set_xlabel("Actual max EMI"); ax.set_ylabel("Predicted")
    plt.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_importance(pipe, path):
    m = pipe.named_steps["model"]
    if not hasattr(m, "feature_importances_"):
        return False
    names = pipe.named_steps["prep"].named_steps["ct"].get_feature_names_out()
    imp = pd.Series(m.feature_importances_, index=names).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    imp.plot.barh(ax=ax, color="#2a6fdb"); ax.set_title("Top 15 feature importances")
    plt.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return True


# ----------------------------------------------------------------- runner
def setup_mlflow(experiment):
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(experiment)


def simple_params(model):
    return {k: v for k, v in model.get_params().items()
            if isinstance(v, (int, float, str, bool)) or v is None}


def run_task(task, models, Xs, ys, classes=None):
    is_clf = task == "classification"
    (Xtr, Xva, Xte), (ytr, yva, yte) = Xs, ys
    metric_fn = clf_metrics if is_clf else reg_metrics
    rows, best = [], None
    for name, model in models.items():
        print(f"\n[{task}] training {name} ...")
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        with mlflow.start_run(run_name=f"{task}_{name}") as run:
            mlflow.set_tags({"task": task, "model_name": name})
            mlflow.log_params({"model": name, "n_train": len(Xtr), **simple_params(model)})
            t0 = time.time(); pipe.fit(Xtr, ytr); fit_s = time.time() - t0
            m = {**metric_fn(pipe, Xva, yva, "val"), **metric_fn(pipe, Xte, yte, "test"),
                 "fit_seconds": fit_s}
            mlflow.log_metrics(m)
            # artifacts
            tag = f"{task}_{name}"
            if is_clf:
                plot_confusion(pipe, Xte, yte, classes, PLOT_DIR / f"{tag}_confusion.png")
                mlflow.log_artifact(str(PLOT_DIR / f"{tag}_confusion.png"), "plots")
            else:
                plot_pred_vs_actual(pipe, Xte, yte, PLOT_DIR / f"{tag}_pred_vs_actual.png")
                mlflow.log_artifact(str(PLOT_DIR / f"{tag}_pred_vs_actual.png"), "plots")
            if plot_importance(pipe, PLOT_DIR / f"{tag}_importance.png"):
                mlflow.log_artifact(str(PLOT_DIR / f"{tag}_importance.png"), "plots")
            mlflow.sklearn.log_model(pipe, artifact_path="model")
            rows.append({"model": name, "run_id": run.info.run_id, **m})
        key = m["val_f1"] if is_clf else -m["val_rmse"]
        if best is None or key > best[0]:
            best = (key, name, pipe, run.info.run_id)
        print("   ", {k: round(v, 4) for k, v in m.items() if k.startswith("val_")})
    res = pd.DataFrame(rows)
    res.to_csv(MODEL_DIR / ("clf_metrics.csv" if is_clf else "reg_metrics.csv"), index=False)
    return best, res


def register_best(run_id, registry_name):
    try:
        mv = mlflow.register_model(f"runs:/{run_id}/model", registry_name)
        MlflowClient().set_registered_model_alias(registry_name, "champion", mv.version)
        print(f"Registered {registry_name} v{mv.version} (alias: champion)")
    except Exception as e:  # registry problems should never stop the pipeline
        print("Model registry step skipped:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(RAW_PATH))
    ap.add_argument("--sample", type=int, default=0, help="use only N rows (quick run)")
    args = ap.parse_args()

    print("Loading", args.data)
    df, report = clean(load_raw(args.data))
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=RANDOM_STATE).reset_index(drop=True)
    (REPORT_DIR / "data_quality.json").write_text(json.dumps(report, indent=2))
    df.sample(min(50_000, len(df)), random_state=RANDOM_STATE).to_csv(SAMPLE_PATH, index=False)
    print(f"Clean rows: {len(df):,}  | report -> reports/data_quality.json")

    tr, va, te = split_data(df)
    le = LabelEncoder().fit(df[TARGET_CLF]); classes = list(le.classes_)
    Xs = (tr[ALL_FEATURES], va[ALL_FEATURES], te[ALL_FEATURES])

    setup_mlflow("EMIPredict_AI")
    (_, clf_name, clf_pipe, clf_run), clf_res = run_task(
        "classification", classifiers(), Xs,
        tuple(le.transform(d[TARGET_CLF]) for d in (tr, va, te)), classes)
    (_, reg_name, reg_pipe, reg_run), reg_res = run_task(
        "regression", regressors(), Xs, tuple(d[TARGET_REG] for d in (tr, va, te)))

    register_best(clf_run, CLF_REGISTRY_NAME)
    register_best(reg_run, REG_REGISTRY_NAME)

    joblib.dump({"pipeline": clf_pipe, "classes": classes, "name": clf_name},
                MODEL_DIR / "classifier.joblib", compress=3)
    joblib.dump({"pipeline": reg_pipe, "name": reg_name},
                MODEL_DIR / "regressor.joblib", compress=3)

    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_rows": int(len(df)), "classes": classes,
        "best_classifier": clf_name, "best_regressor": reg_name,
        "best_clf_metrics": clf_res.set_index("model").loc[clf_name].drop("run_id").to_dict(),
        "best_reg_metrics": reg_res.set_index("model").loc[reg_name].drop("run_id").to_dict(),
        "categories": {c: sorted(df[c].unique().tolist()) for c in CAT_RAW},
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"\nBEST classifier: {clf_name} | BEST regressor: {reg_name}")
    print("Done. Launch MLflow UI:  mlflow ui --backend-store-uri", MLFLOW_URI)


if __name__ == "__main__":
    main()
