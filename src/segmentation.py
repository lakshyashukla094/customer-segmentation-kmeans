"""Customer Segmentation using K-Means clustering.

Run from the project root:  python src/segmentation.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Mall_Customers.csv"
IMG = ROOT / "images"
RES = ROOT / "results"
IMG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]   # used for EDA and profiling
CLUSTER_FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]  # used for K-Means
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    return df


def run_eda(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, FEATURES):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(IMG / "01_distributions.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    df["Gender"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"])
    plt.title("Gender Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(IMG / "02_gender_count.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(df[FEATURES].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(IMG / "03_correlation.png", dpi=150)
    plt.close()


def find_best_k(X_scaled: np.ndarray, k_range=range(2, 11)) -> int:
    inertias, sil = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
        inertias.append(km.inertia_)
        sil.append(silhouette_score(X_scaled, km.labels_))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, marker="o")
    axes[0].set(title="Elbow Method", xlabel="Number of clusters (k)", ylabel="Inertia (WCSS)")
    axes[1].plot(list(k_range), sil, marker="o", color="green")
    axes[1].set(title="Silhouette Score", xlabel="Number of clusters (k)", ylabel="Score")
    plt.tight_layout()
    plt.savefig(IMG / "04_elbow_silhouette.png", dpi=150)
    plt.close()

    best_k = list(k_range)[int(np.argmax(sil))]
    print(f"Best k by silhouette score: {best_k} (score = {max(sil):.3f})")
    return best_k


def name_segment(row: pd.Series, df: pd.DataFrame) -> str:
    inc, spd = row["Annual Income (k$)"], row["Spending Score (1-100)"]
    inc_lo, inc_hi = df["Annual Income (k$)"].quantile([0.33, 0.67])
    spd_lo, spd_hi = df["Spending Score (1-100)"].quantile([0.33, 0.67])
    inc_lvl = "Low" if inc < inc_lo else ("High" if inc > inc_hi else "Mid")
    spd_lvl = "Low" if spd < spd_lo else ("High" if spd > spd_hi else "Mid")
    if inc_lvl == "Mid" and spd_lvl == "Mid":
        return "Standard (Average Income, Average Spend)"
    if inc_lvl == "High" and spd_lvl == "High":
        return "Premium (High Income, High Spend)"
    if inc_lvl == "High":
        return "Careful (High Income, Low Spend)"
    if spd_lvl == "High":
        return "Impulsive (Low Income, High Spend)"
    return "Budget (Low Income, Low Spend)"


def main() -> None:
    df = load_data()
    run_eda(df)

    X = df[CLUSTER_FEATURES]
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    k = find_best_k(X_scaled)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled)
    df["Cluster"] = km.labels_

    # Cluster profile and business names
    profile = df.groupby("Cluster")[FEATURES].mean().round(1)
    profile["Customers"] = df["Cluster"].value_counts().sort_index()
    names = profile.apply(lambda r: name_segment(r, df), axis=1)
    # make names unique if two clusters land in the same quadrant
    seen = {}
    for idx, n in names.items():
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[idx] = f"{n} #{seen[n]}"
    profile["Segment"] = names
    print("\nCluster profile:\n", profile.to_string())
    profile.to_csv(RES / "cluster_profile.csv")

    df["Segment"] = df["Cluster"].map(profile["Segment"])
    df.to_csv(RES / "customers_with_clusters.csv", index=False)

    # Visuals
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=CLUSTER_FEATURES)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)",
                    hue="Segment", palette="deep", s=70)
    plt.scatter(centers["Annual Income (k$)"], centers["Spending Score (1-100)"],
                c="black", marker="X", s=250, label="Centroids")
    plt.title("Customer Segments (Income vs Spending Score)")
    plt.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(IMG / "05_segments_income_spending.png", dpi=150)
    plt.close()

    pcs = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=df["Segment"], palette="deep", s=70)
    plt.title("Clusters in 2D (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.tight_layout()
    plt.savefig(IMG / "06_pca_clusters.png", dpi=150)
    plt.close()

    ax = profile[FEATURES].plot(kind="bar", figsize=(8, 5))
    ax.set_title("Average Feature Values per Cluster")
    ax.set_xlabel("Cluster")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(IMG / "07_cluster_profile.png", dpi=150)
    plt.close()

    print(f"\nDone. Charts saved in {IMG}, tables in {RES}")


if __name__ == "__main__":
    main()
