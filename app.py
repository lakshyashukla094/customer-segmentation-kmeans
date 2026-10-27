"""Customer Segmentation web app (Streamlit).

Run locally:  streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "Mall_Customers.csv"
INCOME = "Annual Income (k$)"
SPEND = "Spending Score (1-100)"
N_CLUSTERS = 5
SEED = 42

SEGMENT_INFO = {
    "Premium": (
        "High income and high spending. Your best customers.",
        "VIP loyalty program, early access to new products, premium services.",
    ),
    "Careful": (
        "High income but low spending. Big potential that is not being used yet.",
        "Personalised offers and reminders; find out what would make them spend more.",
    ),
    "Impulsive": (
        "Low income but high spending. Young, enthusiastic buyers.",
        "Flash sales, limited-time deals, bundles and instalment (EMI) options.",
    ),
    "Standard": (
        "Average income and average spending. The largest group.",
        "Regular promotions, upselling and cross-selling.",
    ),
    "Budget": (
        "Low income and low spending. Price-sensitive customers.",
        "Discount coupons and low-cost entry products; keep marketing spend low.",
    ),
}
COLORS = {
    "Premium": "#d62728",
    "Careful": "#ff7f0e",
    "Impulsive": "#2ca02c",
    "Standard": "#1f77b4",
    "Budget": "#9467bd",
}


def segment_key(df: pd.DataFrame, income: float, spend: float) -> str:
    """Name a cluster from its average income and spending (same logic as src/segmentation.py)."""
    inc_lo, inc_hi = df[INCOME].quantile([0.33, 0.67])
    spd_lo, spd_hi = df[SPEND].quantile([0.33, 0.67])
    inc = "Low" if income < inc_lo else ("High" if income > inc_hi else "Mid")
    spd = "Low" if spend < spd_lo else ("High" if spend > spd_hi else "Mid")
    if inc == "Mid" and spd == "Mid":
        return "Standard"
    if inc == "High" and spd == "High":
        return "Premium"
    if inc == "High":
        return "Careful"
    if spd == "High":
        return "Impulsive"
    return "Budget"


@st.cache_resource
def load_model():
    """Load the data, scale it, fit K-Means and name every cluster."""
    df = pd.read_csv(DATA_PATH)
    scaler = StandardScaler().fit(df[[INCOME, SPEND]])
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit(
        scaler.transform(df[[INCOME, SPEND]])
    )
    df["Cluster"] = km.labels_
    profile = df.groupby("Cluster")[["Age", INCOME, SPEND]].mean().round(1)
    profile["Customers"] = df["Cluster"].value_counts().sort_index()
    profile["Segment"] = [
        segment_key(df, row[INCOME], row[SPEND]) for _, row in profile.iterrows()
    ]
    df["Segment"] = df["Cluster"].map(profile["Segment"])
    return df, scaler, km, profile


def predict_segment(scaler, km, profile, income: float, spend: float) -> str:
    x = scaler.transform(pd.DataFrame([[income, spend]], columns=[INCOME, SPEND]))
    return profile.loc[int(km.predict(x)[0]), "Segment"]


def make_chart(df: pd.DataFrame, income: float, spend: float, segment: str):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, group in df.groupby("Segment"):
        ax.scatter(group[INCOME], group[SPEND], s=45, alpha=0.75,
                   color=COLORS.get(name, "gray"), label=name)
    ax.scatter([income], [spend], s=350, marker="*", color="black",
               edgecolor="white", linewidth=1.5, label="Your customer", zorder=5)
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title(f"Customer segments (new customer: {segment})")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")
    st.title("🛍️ Customer Segmentation using K-Means")
    st.caption(
        "Enter a customer's annual income and spending score to see which "
        "segment they belong to and how to market to them."
    )

    df, scaler, km, profile = load_model()

    st.sidebar.header("New customer")
    income = st.sidebar.slider("Annual income (k$)", 15, 140, 60)
    spend = st.sidebar.slider("Spending score (1-100)", 1, 100, 50)
    st.sidebar.markdown("---")
    st.sidebar.write(
        "Model: K-Means with k = 5, trained on 200 mall customers "
        "(Kaggle: Mall Customer Segmentation Data)."
    )

    segment = predict_segment(scaler, km, profile, income, spend)
    description, strategy = SEGMENT_INFO[segment]
    size = int((df["Segment"] == segment).sum())

    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Prediction")
        st.metric("Predicted segment", segment)
        st.write(description)
        st.success("Suggested strategy: " + strategy)
        st.write(f"{size} of {len(df)} customers ({size / len(df):.0%}) are in this segment.")
    with right:
        st.pyplot(make_chart(df, income, spend, segment))

    tab1, tab2 = st.tabs(["Segment overview", "Dataset sample"])
    with tab1:
        overview = profile.rename(columns={"Age": "Avg age", INCOME: "Avg income (k$)",
                                           SPEND: "Avg spending score"})
        st.dataframe(overview.set_index("Segment"))
    with tab2:
        st.dataframe(df.drop(columns=["Cluster"]).head(20))


if __name__ == "__main__":
    main()
