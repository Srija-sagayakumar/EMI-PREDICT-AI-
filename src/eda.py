"""Exploratory data analysis -> PNG charts + text summary in reports/eda/.
Usage: python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import RAW_PATH, REPORT_DIR, TARGET_CLF, TARGET_REG
from .features import add_features
from .preprocess import clean, load_raw

OUT = REPORT_DIR / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def save(name):
    plt.tight_layout(); plt.savefig(OUT / name, dpi=110); plt.close()


def main():
    df, _ = clean(load_raw(RAW_PATH))
    df = add_features(df)
    sns.set_theme(style="whitegrid")

    df[TARGET_CLF].value_counts().plot.bar(color="#2a6fdb", title="EMI eligibility distribution")
    save("01_eligibility_distribution.png")

    pd_ct = df.groupby(["emi_scenario", TARGET_CLF]).size().unstack(fill_value=0)
    (pd_ct.div(pd_ct.sum(1), axis=0) * 100).plot.bar(stacked=True, figsize=(9, 5),
                                                   title="Eligibility % by EMI scenario")
    save("02_eligibility_by_scenario.png")

    plt.figure(figsize=(7, 5)); sns.boxplot(data=df, x=TARGET_CLF, y="credit_score")
    plt.title("Credit score by eligibility"); save("03_credit_vs_eligibility.png")

    plt.figure(figsize=(7, 5)); sns.boxplot(data=df, x=TARGET_CLF, y="debt_to_income")
    plt.title("Debt-to-income by eligibility"); save("04_dti_vs_eligibility.png")

    plt.figure(figsize=(7, 5))
    s = df.sample(min(5000, len(df)), random_state=1)
    sns.scatterplot(data=s, x="monthly_salary", y=TARGET_REG, hue=TARGET_CLF, s=12, alpha=.6)
    plt.title("Salary vs max monthly EMI"); save("05_salary_vs_max_emi.png")

    plt.figure(figsize=(12, 9))
    sns.heatmap(df.select_dtypes("number").corr(), cmap="coolwarm", center=0)
    plt.title("Correlation matrix"); save("06_correlation.png")

    sns.countplot(data=df, x="education", hue=TARGET_CLF); plt.xticks(rotation=15)
    plt.title("Education vs eligibility"); save("07_education.png")

    df.describe().T.to_csv(OUT / "statistical_summary.csv")
    print("EDA charts saved to", OUT)


if __name__ == "__main__":
    main()
