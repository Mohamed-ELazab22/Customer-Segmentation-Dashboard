from contextlib import contextmanager
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==========================================
# Page Config
# ==========================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "kmeans_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
PCA_PATH = BASE_DIR / "pca.pkl"
DATA_PATH = BASE_DIR / "Wholesale customers data.csv"
CSS_PATH = BASE_DIR / "style.css"


# ==========================================
# Load CSS
# ==========================================

def load_css():
    if CSS_PATH.exists():
        with open(CSS_PATH, encoding="utf-8") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )


load_css()


# ==========================================
# Load Models
# ==========================================

@st.cache_resource
def load_models():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)

    return model, scaler, pca


model, scaler, pca = load_models()


# ==========================================
# Load Dataset
# ==========================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)

# ==========================================
# Labels
# ==========================================

REGION_MAP = {
    1: "Lisbon",
    2: "Oporto",
    3: "Other Region"
}

CHANNEL_MAP = {
    1: "Horeca",
    2: "Retail"
}

CLUSTER_MAP = {
    0: "Retail & Supermarket Customers",
    1: "Hotels, Restaurants & Cafés"
}
df = load_data()
df["Region"] = df["Region"].map(REGION_MAP)
df["Channel"] = df["Channel"].map(CHANNEL_MAP)


FEATURES = [
    "Fresh",
    "Milk",
    "Grocery",
    "Frozen",
    "Detergents_Paper",
    "Delicassen"
]




# ==========================================
# Reusable UI Components
# ==========================================

def metric_card(icon, title, value, detail=""):
    """Render a premium metric card without changing application state."""
    detail_html = (
        f'<div style="color:#b6b6b6;font-size:0.82rem;margin-top:7px;">'
        f'{escape(str(detail))}</div>'
        if detail else ""
    )
    st.markdown(
        f"""
        <div class="metric-box">
            <div style="font-size:1.35rem;margin-bottom:7px;">{icon}</div>
            <div class="metric-title">{escape(str(title))}</div>
            <div class="metric-value">{escape(str(value))}</div>
            {detail_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title, subtitle="", eyebrow="ANALYTICS"):
    """Render a consistent page or section heading."""
    subtitle_html = (
        f'<p style="margin:7px 0 0;color:#b6b6b6;font-size:0.98rem;">'
        f'{escape(str(subtitle))}</p>'
        if subtitle else ""
    )
    st.markdown(
        f"""
        <div style="margin:8px 0 18px;">
            <div style="color:#b99878;font-size:0.72rem;font-weight:700;letter-spacing:0.14em;">{escape(str(eyebrow))}</div>
            <h2 style="margin:5px 0 0;color:#ffffff;font-size:1.8rem;">{escape(str(title))}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True
    )


def hero_panel(title, description, badge="CUSTOMER INTELLIGENCE"):
    """Render the dashboard hero while reusing the existing .hero CSS class."""
    st.markdown(
        f"""
        <div class="hero">
            <div style="position:relative;z-index:1;max-width:760px;">
                <div style="display:inline-block;padding:6px 10px;border:1px solid rgba(255,255,255,.35);border-radius:999px;font-size:.72rem;font-weight:700;letter-spacing:.1em;">{escape(str(badge))}</div>
                <h1>{escape(str(title))}</h1>
                <p>{escape(str(description))}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


@contextmanager
def content_card(title, subtitle="", eyebrow="OVERVIEW"):
    """Create a titled Streamlit container with the dashboard card treatment."""
    section_title(title, subtitle, eyebrow)
    with st.container(border=True):
        yield


def info_card(title, content, icon="✦"):
    """Render concise, reusable information cards for narrative content."""
    st.markdown(
        f"""
        <div class="card" style="height:100%;margin-top:0;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <span style="color:#b99878;font-size:1.25rem;">{icon}</span>
                <span style="color:#ffffff;font-weight:700;font-size:1.02rem;">{escape(str(title))}</span>
            </div>
            <div style="color:#b6b6b6;line-height:1.65;font-size:.93rem;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def apply_chart_theme(fig, height=None):
    """Apply a shared polished Plotly presentation style without changing chart data."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1C1F27",
        plot_bgcolor="#1C1F27",
        font=dict(color="#F7F7F7", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=62, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


# ==========================================
# Sidebar
# ==========================================

PAGES = ["Dashboard", "Prediction", "Visualization", "Dataset", "About"]
PAGE_ICONS = {
    "Dashboard": "▦",
    "Prediction": "◎",
    "Visualization": "◌",
    "Dataset": "▤",
    "About": "i"
}

with st.sidebar:

    st.title("Customer Intelligence")
    st.caption("K-Means segmentation workspace")
    st.markdown(
        "<div style='color:#b99878;font-size:.75rem;font-weight:700;letter-spacing:.12em;margin:22px 0 4px;'>WORKSPACE</div>",
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        PAGES,
        format_func=lambda item: f"{PAGE_ICONS[item]}  {item}",
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Wholesale customer behavior analysis")


# ==========================================
# Dashboard
# ==========================================

if page == "Dashboard":

    hero_panel(
        "Customer Segmentation Dashboard",
        "Turn wholesale purchase behavior into clear, actionable customer segments. Explore the data, predict new customer groups, and compare spending patterns in one focused workspace."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("◉", "Customers", f"{len(df):,}", "Wholesale records")

    with col2:
        metric_card("◫", "Features", len(FEATURES), "Spending categories")

    with col3:
        metric_card("◌", "Clusters", model.n_clusters, "K-Means segments")

    with col4:
        metric_card("⌁", "Algorithm", "K-Means", "Unsupervised learning")

    with content_card(
        "Dataset snapshot",
        "A quick look at the most recent customer records in the source dataset.",
        "DATA EXPLORATION"
    ):
        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )

    with content_card(
        "Spending profile summary",
        "Descriptive statistics for the annual customer spending categories.",
        "DATA QUALITY"
    ):
        st.dataframe(
            df.describe(),
            use_container_width=True
        )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with content_card(
            "Channel distribution",
            "Customer composition by sales channel.",
            "DISTRIBUTION"
        ):
            fig = px.pie(
                df,
                names="Channel",
                title="Customers by Channel",
                template="plotly_dark"
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            apply_chart_theme(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        with content_card(
            "Region distribution",
            "Customer volume across the available regions.",
            "DISTRIBUTION"
        ):
            fig = px.bar(
                df["Region"].value_counts().reset_index(),
                x="Region",
                y="count",
                color="Region",
                template="plotly_dark",
                title="Customers by Region"
            )
            fig.update_layout(showlegend=False)
            apply_chart_theme(fig, height=360)
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# Prediction Page
# ==========================================

elif page == "Prediction":

    hero_panel(
        "Customer Cluster Prediction",
        "Enter annual category spend to place a customer within the existing K-Means segmentation model.",
        "PREDICTIVE ANALYTICS"
    )

    with content_card(
        "Customer spending profile",
        "All figures represent annual spend. Update the values, then run the prediction.",
        "INPUT FORM"
    ):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            fresh = st.number_input(
                "Fresh",
                min_value=0.0,
                value=12000.0,
                step=100.0
            )

            milk = st.number_input(
                "Milk",
                min_value=0.0,
                value=6000.0,
                step=100.0
            )

            grocery = st.number_input(
                "Grocery",
                min_value=0.0,
                value=8000.0,
                step=100.0
            )

        with col2:
            frozen = st.number_input(
                "Frozen",
                min_value=0.0,
                value=2500.0,
                step=100.0
            )

            detergents = st.number_input(
                "Detergents Paper",
                min_value=0.0,
                value=2000.0,
                step=100.0
            )

            delicassen = st.number_input(
                "Delicassen",
                min_value=0.0,
                value=1500.0,
                step=100.0
            )

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        predict_clicked = st.button(
            "Predict Customer Cluster",
            use_container_width=True
        )

    if predict_clicked:

        customer = pd.DataFrame(
            [[
                1,
                np.log1p(fresh),
                np.log1p(milk),
                np.log1p(grocery),
                np.log1p(frozen),
                np.log1p(detergents),
                np.log1p(delicassen),
            ]],
            columns=[
                "Channel",
                "Fresh",
                "Milk",
                "Grocery",
                "Frozen",
                "Detergents_Paper",
                "Delicassen",
            ]
        )

        scaled = scaler.transform(customer)

        cluster = model.predict(scaled)[0]
        cluster_name = CLUSTER_MAP.get(cluster, f"Cluster {cluster}")

        distance = model.transform(scaled).min()

        confidence = 100 / (1 + distance)

        with content_card(
            "Prediction result",
            "The profile has been evaluated against the trained cluster centers.",
            "MODEL OUTPUT"
        ):
            st.success(f"Predicted Cluster: {cluster_name}")

            c1, c2, c3 = st.columns(3)

            with c1:
                metric_card("◌", "Cluster", cluster_name, "Assigned segment")
            with c2:
                metric_card("↔", "Distance", f"{distance:.3f}", "Nearest center")
            with c3:
                metric_card("✦", "Confidence", f"{confidence:.1f}%", "Distance-based score")

        center = scaler.inverse_transform(
            model.cluster_centers_[cluster].reshape(1, -1)
        )[0]

        center[1:] = np.expm1(center[1:])

        compare = pd.DataFrame({
            "Feature": FEATURES,
            "Customer": [
                fresh,
                milk,
                grocery,
                frozen,
                detergents,
                delicassen
            ],
            "Cluster Center": [
                center[1],
                center[2],
                center[3],
                center[4],
                center[5],
                center[6]
            ]
        })

        insight_col, detail_col = st.columns([1.05, 1.95], gap="large")
        with insight_col:
            leading_feature = compare.loc[compare["Customer"].idxmax(), "Feature"]
            customer_total = compare["Customer"].sum()
            info_card(
                "Customer insights",
                f"<strong style='color:#ffffff;'>Primary spend:</strong> {escape(str(leading_feature))}<br>"
                f"<strong style='color:#ffffff;'>Total annual spend:</strong> {customer_total:,.0f}<br><br>"
                "Use the comparison to understand how this profile differs from the center of its assigned segment.",
                "✦"
            )

        with detail_col:
            with content_card(
                "Profile comparison",
                "Customer spending benchmarked against the assigned cluster center.",
                "SPENDING ANALYSIS"
            ):
                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=compare["Feature"],
                        y=compare["Customer"],
                        name="Customer",
                        marker_color="#B99878"
                    )
                )

                fig.add_trace(
                    go.Bar(
                        x=compare["Feature"],
                        y=compare["Cluster Center"],
                        name="Cluster Center",
                        marker_color="#5C8D89"
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    barmode="group",
                    height=500,
                    title="Customer vs Cluster Center"
                )
                apply_chart_theme(fig, height=460)

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with content_card(
            "Feature-level comparison",
            "The exact values used for the visual benchmark above.",
            "DETAIL TABLE"
        ):
            st.dataframe(
                compare,
                use_container_width=True,
                hide_index=True
            )

        st.session_state["prediction"] = pca.transform(scaled)
        st.session_state["cluster"] = cluster


# ==========================================
# Visualization Page
# ==========================================

elif page == "Visualization":

    hero_panel(
        "Cluster Visualization",
        "Explore how the trained customer segments separate after PCA dimensionality reduction.",
        "SEGMENT EXPLORER"
    )

    X = df.copy()

    X = X[
        [
            "Channel",
            "Fresh",
            "Milk",
            "Grocery",
            "Frozen",
            "Detergents_Paper",
            "Delicassen",
        ]
    ]

    X[FEATURES] = np.log1p(X[FEATURES])

    X_scaled = scaler.transform(X)

    labels = model.predict(X_scaled)

    X_pca = pca.transform(X_scaled)

    plot_df = pd.DataFrame({
      "PCA1": X_pca[:, 0],
      "PCA2": X_pca[:, 1],
      "Cluster": [CLUSTER_MAP[i] for i in labels]
     })

    with content_card(
        "Customer segment map",
        "Each point is a customer. Markers identify the learned cluster centers and any recent prediction.",
        "PCA PROJECTION"
    ):
        fig = px.scatter(
            plot_df,
            x="PCA1",
            y="PCA2",
            color="Cluster",
            template="plotly_dark",
            title="Customer Segments",
            height=650
        )

        centers = pca.transform(model.cluster_centers_)

        fig.add_trace(
            go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode="markers+text",
                text=[CLUSTER_MAP[i] for i in range(model.n_clusters)],
                textposition="top center",
                marker=dict(
                    size=18,
                    symbol="x",
                    color="white"
                ),
                name="Cluster Centers"
            )
        )

        if "prediction" in st.session_state:

            customer = st.session_state["prediction"]

            cluster = st.session_state["cluster"]
            cluster_name = CLUSTER_MAP.get(cluster, f"Cluster {cluster}")

            fig.add_trace(
                go.Scatter(
                    x=[customer[0][0]],
                    y=[customer[0][1]],
                    mode="markers",
                    marker=dict(
                        size=22,
                        color="red",
                        symbol="star"
                    ),
                    name=f"Customer ({cluster_name})"
                )
            )

        fig.update_traces(
            marker=dict(line=dict(width=0.5, color="rgba(255,255,255,.45)")),
            selector=dict(mode="markers")
        )
        apply_chart_theme(fig, height=650)

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ==========================================
# Dataset Page
# ==========================================

elif page == "Dataset":

    hero_panel(
        "Dataset Overview",
        "Review the structure, completeness, and relationships within the wholesale customer dataset.",
        "DATA EXPLORATION"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("▦", "Rows", df.shape[0], "Customer records")
    with c2:
        metric_card("▤", "Columns", df.shape[1], "Available fields")
    with c3:
        metric_card("◫", "Numeric fields", df.select_dtypes(include=np.number).shape[1], "Analysis-ready")
    with c4:
        metric_card("✓", "Missing values", df.isna().sum().sum(), "Data completeness")

    with content_card(
        "Dataset preview",
        "The first ten records from the original dataset.",
        "SOURCE DATA"
    ):
        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )

    dataset_col1, dataset_col2 = st.columns([1.55, 1], gap="large")
    with dataset_col1:
        with content_card(
            "Summary statistics",
            "Distribution and range of each numerical field.",
            "DESCRIPTIVE ANALYTICS"
        ):
            st.dataframe(
                df.describe(),
                use_container_width=True
            )

    with dataset_col2:
        with content_card(
            "Missing values",
            "Null-count audit for every dataset field.",
            "DATA QUALITY"
        ):
            missing_values = df.isna().sum().rename("Missing values").to_frame()
            st.dataframe(
                missing_values,
                use_container_width=True
            )

    with content_card(
        "Correlation matrix",
        "Relationships between the numerical variables in the dataset.",
        "RELATIONSHIP ANALYSIS"
    ):
        corr = df.corr(numeric_only=True)

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="Blues",
            template="plotly_dark",
            aspect="auto"
        )
        fig.update_layout(coloraxis_colorbar=dict(title="Correlation"))
        apply_chart_theme(fig, height=560)

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ==========================================
# About Page
# ==========================================

elif page == "About":

    hero_panel(
        "About the Project",
        "A focused analytics dashboard for understanding wholesale customer behavior through K-Means segmentation.",
        "PROJECT OVERVIEW"
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        info_card(
            "Dataset",
            "The Wholesale Customers Dataset captures annual customer spending across six categories: Fresh, Milk, Grocery, Frozen, Detergents Paper, and Delicassen.",
            "▤"
        )
    with col2:
        info_card(
            "Data preprocessing",
            "The existing workflow applies log transformation with <strong style='color:#ffffff;'>np.log1p</strong>, then StandardScaler and Principal Component Analysis (PCA).",
            "◫"
        )

    col3, col4 = st.columns(2, gap="large")
    with col3:
        info_card(
            "Machine learning model",
            "Customer groups are produced with <strong style='color:#ffffff;'>K-Means Clustering</strong>. The number of clusters was selected with the Elbow Method and validated with Silhouette Score.",
            "◌"
        )
    with col4:
        info_card(
            "Dashboard capabilities",
            "Explore customer data, predict cluster membership for a new profile, inspect a PCA segment map, and review correlations and dataset quality.",
            "✦"
        )

    section_title("Technology stack", "Built for clear, interactive customer segmentation analysis.", "IMPLEMENTATION")
    tech_cols = st.columns(4)
    technologies = [
        ("Python", "Core application language"),
        ("Streamlit", "Interactive dashboard"),
        ("Scikit-Learn", "Segmentation pipeline"),
        ("Plotly", "Interactive charts")
    ]
    for column, (name, detail) in zip(tech_cols, technologies):
        with column:
            metric_card("◆", name, "", detail)

    st.markdown(
        "<div style='text-align:center;color:#b6b6b6;padding:28px 0 8px;'>"
        "Developed by <span style='color:#ffffff;font-weight:700;'>Mohamed Elazab</span> · "
        "Faculty of Computers &amp; Artificial Intelligence</div>",
        unsafe_allow_html=True
    )
