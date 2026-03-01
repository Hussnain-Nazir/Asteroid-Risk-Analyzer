"""
app.py
------
Asteroid Risk Analyzer — Streamlit Application
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data
from src.preprocessing import clean_and_prepare, get_feature_names
from src.model import train_model, predict_single

# ─────────────────────────────────────────────
# THEME CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌍 Asteroid Risk Analyzer",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_BG = "#0d1117"
CARD_BG = "#161b22"
ACCENT = "#00ffe7"
ACCENT2 = "#ff6b35"
TEXT = "#e6edf3"
MUTED = "#8b949e"

SPACE_CSS = f"""
<style>
    /* ── Global ── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {DARK_BG};
        color: {TEXT};
    }}
    [data-testid="stSidebar"] {{
        background-color: #010409;
        border-right: 1px solid #30363d;
    }}
    /* ── Headers ── */
    h1, h2, h3, h4 {{
        color: {ACCENT} !important;
        font-family: 'Courier New', monospace;
    }}
    /* ── Metrics ── */
    [data-testid="metric-container"] {{
        background: {CARD_BG};
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px;
    }}
    [data-testid="stMetricLabel"] {{ color: {MUTED} !important; }}
    [data-testid="stMetricValue"] {{ color: {ACCENT} !important; }}
    /* ── Buttons ── */
    .stButton > button {{
        background: linear-gradient(135deg, #0d4f4f, #1a8080);
        color: {ACCENT};
        border: 1px solid {ACCENT};
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        transition: all 0.2s;
    }}
    .stButton > button:hover {{
        background: {ACCENT};
        color: {DARK_BG};
    }}
    /* ── Sliders & Inputs ── */
    .stSlider, .stNumberInput {{ color: {TEXT}; }}
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] {{
        color: {MUTED};
        font-family: 'Courier New', monospace;
    }}
    .stTabs [aria-selected="true"] {{
        color: {ACCENT} !important;
        border-bottom: 2px solid {ACCENT};
    }}
    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {{ background: {CARD_BG}; }}
    /* ── Dividers ── */
    hr {{ border-color: #30363d; }}
    /* ── Alert boxes ── */
    .hazard-box {{
        background: linear-gradient(135deg, #3d0000, #700000);
        border: 2px solid #ff4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }}
    .safe-box {{
        background: linear-gradient(135deg, #003d1a, #006b2e);
        border: 2px solid #00ff7f;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }}
    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {DARK_BG}; }}
    ::-webkit-scrollbar-thumb {{ background: #30363d; border-radius: 3px; }}
</style>
"""
st.markdown(SPACE_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": CARD_BG,
    "axes.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": TEXT,
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "axes.titlecolor": ACCENT,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="🛸 Loading asteroid data...")
def get_data():
    raw = load_data()
    clean = clean_and_prepare(raw)
    return raw, clean


@st.cache_resource(show_spinner="🤖 Training model...")
def get_model(df_hash: int):
    _, clean = get_data()
    return train_model(clean)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## ☄️ Asteroid Risk Analyzer")
    st.markdown(f"<span style='color:{MUTED}'>Near-Earth Object Hazard Classification</span>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Data Analysis", "🤖 Model Performance", "🔭 Risk Prediction"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(f"<small style='color:{MUTED}'>Model: Random Forest</small>", unsafe_allow_html=True)

# Load data
try:
    raw_df, clean_df = get_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"❌ Could not load data: {e}\n\nCheck your NASA API key in `src/data_loader.py`.")

if data_loaded:
    model, metrics = get_model(hash(str(clean_df.shape)))
    feature_names = metrics["feature_names"]


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("# 🌍 Asteroid Risk Analyzer")
    st.markdown(f"<p style='color:{MUTED}; font-size:1.1em'>Machine Learning Project | Binary Classification of Near-Earth Objects</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🎯 Project Overview")
        st.markdown("""
        This project uses **real NASA data** to classify asteroids as either:
        - 🔴 **Potentially Hazardous** — close approach within 0.05 AU & diameter > 140m
        - 🟢 **Safe** — no significant threat to Earth

        A **Random Forest Classifier** is trained on physical and orbital features to
        automate hazard assessment, demonstrate class imbalance handling, and provide
        an interactive risk prediction interface.
        """)

        st.markdown("### 📡 Data Source")
        st.markdown("""
        **NASA Near Earth Object Web Service (NeoWs)**
        - Real-time asteroid tracking data from NASA's Jet Propulsion Laboratory
        - Updated daily with close approach information
        - API endpoint: `https://api.nasa.gov/neo/rest/v1/feed`
        - Data is fetched automatically on first launch and cached locally as CSV
        """)

    with col2:
        st.markdown("### 📈 Dataset Stats")
        if data_loaded:
            total = len(clean_df)
            hazardous = clean_df["is_potentially_hazardous_asteroid"].sum()
            safe = total - hazardous
            st.metric("Total Asteroids", f"{total:,}")
            st.metric("🔴 Hazardous", f"{hazardous:,}", f"{hazardous/total:.1%}")
            st.metric("🟢 Safe", f"{safe:,}", f"{safe/total:.1%}")

    st.markdown("---")
    st.markdown("### 🧬 Feature Engineering")
    feature_descriptions = {
        "absolute_magnitude_h": "Intrinsic brightness — proxy for asteroid size",
        "diameter_min_km": "Estimated minimum diameter in kilometers",
        "diameter_max_km": "Estimated maximum diameter in kilometers",
        "diameter_mean_km": "Derived: average of min/max diameter",
        "relative_velocity_km_s": "Speed relative to Earth at closest approach",
        "miss_distance_km": "Closest approach distance from Earth",
        "orbital_eccentricity": "How elliptical the orbit is (0=circle, 1=parabola)",
    }
    feat_df = pd.DataFrame(
        [(k, v) for k, v in feature_descriptions.items()],
        columns=["Feature", "Description"]
    )
    st.dataframe(feat_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PAGE: DATA ANALYSIS
# ─────────────────────────────────────────────
elif page == "📊 Data Analysis" and data_loaded:
    st.markdown("# 📊 Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📋 Dataset Preview", "📈 Visualizations", "🔥 Correlations"])

    with tab1:
        st.markdown("### Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(clean_df):,}")
        col2.metric("Features", len(feature_names))
        col3.metric("Hazardous Rate", f"{clean_df['is_potentially_hazardous_asteroid'].mean():.1%}")

        st.dataframe(clean_df.head(50), use_container_width=True)

        st.markdown("### Summary Statistics")
        st.dataframe(clean_df[feature_names].describe().T.round(4), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        # 1. Class distribution
        with col1:
            st.markdown("#### 1. Class Distribution")
            fig, ax = plt.subplots(figsize=(5, 4))
            counts = clean_df["is_potentially_hazardous_asteroid"].value_counts()
            bars = ax.bar(["Safe", "Hazardous"], [counts.get(0, 0), counts.get(1, 0)],
                         color=["#00ff7f", "#ff4444"], edgecolor="#30363d", linewidth=1.5)
            ax.set_ylabel("Count")
            ax.set_title("Asteroid Classification Balance")
            for bar, count in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f"{count:,}", ha="center", va="bottom", color=TEXT, fontsize=10)
            ax.grid(axis="y", alpha=0.4)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # 2. Diameter vs hazard
        with col2:
            st.markdown("#### 2. Diameter vs Hazard Class")
            fig, ax = plt.subplots(figsize=(5, 4))
            safe_d = clean_df[clean_df["is_potentially_hazardous_asteroid"] == 0]["diameter_mean_km"].dropna()
            haz_d = clean_df[clean_df["is_potentially_hazardous_asteroid"] == 1]["diameter_mean_km"].dropna()
            ax.violinplot([safe_d.clip(0, 3), haz_d.clip(0, 3)], positions=[0, 1],
                         showmedians=True)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Safe", "Hazardous"])
            ax.set_ylabel("Mean Diameter (km)")
            ax.set_title("Diameter Distribution by Class")
            ax.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        col3, col4 = st.columns(2)

        # 3. Miss distance histogram
        with col3:
            st.markdown("#### 3. Miss Distance (log scale)")
            fig, ax = plt.subplots(figsize=(5, 4))
            safe_m = clean_df[clean_df["is_potentially_hazardous_asteroid"] == 0]["miss_distance_km"].dropna()
            haz_m = clean_df[clean_df["is_potentially_hazardous_asteroid"] == 1]["miss_distance_km"].dropna()
            ax.hist(safe_m, bins=40, alpha=0.6, color="#00ff7f", label="Safe", log=True)
            ax.hist(haz_m, bins=40, alpha=0.6, color="#ff4444", label="Hazardous", log=True)
            ax.set_xlabel("Miss Distance (km)")
            ax.set_ylabel("Count (log)")
            ax.set_title("Miss Distance Distribution")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # 4. Velocity vs miss distance
        with col4:
            st.markdown("#### 4. Velocity vs Miss Distance")
            fig, ax = plt.subplots(figsize=(5, 4))
            safe_data = clean_df[clean_df["is_potentially_hazardous_asteroid"] == 0]
            haz_data = clean_df[clean_df["is_potentially_hazardous_asteroid"] == 1]
            ax.scatter(safe_data["relative_velocity_km_s"], safe_data["miss_distance_km"] / 1e6,
                      alpha=0.4, s=15, color="#00ff7f", label="Safe")
            ax.scatter(haz_data["relative_velocity_km_s"], haz_data["miss_distance_km"] / 1e6,
                      alpha=0.6, s=20, color="#ff4444", label="Hazardous")
            ax.set_xlabel("Velocity (km/s)")
            ax.set_ylabel("Miss Distance (million km)")
            ax.set_title("Velocity vs Miss Distance")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with tab3:
        st.markdown("#### 5. Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        num_cols = [c for c in feature_names if c in clean_df.columns] + ["is_potentially_hazardous_asteroid"]
        corr = clean_df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                   center=0, ax=ax, linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   annot_kws={"size": 8})
        ax.set_title("Feature Correlation Matrix", pad=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ─────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "🤖 Model Performance" and data_loaded:
    st.markdown("# 🤖 Model Performance")
    st.markdown(f"<span style='color:{MUTED}'>Random Forest Classifier</span>", unsafe_allow_html=True)
    st.markdown("---")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("Precision", f"{metrics['precision']:.3f}")
    col3.metric("Recall", f"{metrics['recall']:.3f}")
    col4.metric("F1-Score", f"{metrics['f1']:.3f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = metrics["confusion_matrix"]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Hazardous"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.title.set_color(ACCENT)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ROC Curve
    with col2:
        st.markdown("#### ROC Curve")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(metrics["roc_fpr"], metrics["roc_tpr"],
                color=ACCENT, lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], color="#30363d", lw=1, linestyle="--", label="Random")
        ax.fill_between(metrics["roc_fpr"], metrics["roc_tpr"], alpha=0.1, color=ACCENT)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    col3, col4 = st.columns(2)

    # Feature Importance
    with col3:
        st.markdown("#### Feature Importance")
        fig, ax = plt.subplots(figsize=(5, 4))
        feat_imp = pd.Series(metrics["feature_importances"], index=metrics["feature_names"]).sort_values()
        colors = [ACCENT if i == feat_imp.index[-1] else "#1a6b6b" for i in feat_imp.index]
        bars = ax.barh(feat_imp.index, feat_imp.values, color=colors, edgecolor="#30363d")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Prediction Probability Distribution
    with col4:
        st.markdown("#### Prediction Probability Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        y_test = metrics["y_test"]
        y_proba = metrics["y_proba"]
        ax.hist(y_proba[y_test == 0], bins=30, alpha=0.6, color="#00ff7f", label="Safe")
        ax.hist(y_proba[y_test == 1], bins=30, alpha=0.6, color="#ff4444", label="Hazardous")
        ax.axvline(0.5, color="white", linestyle="--", linewidth=1, label="Threshold 0.5")
        ax.set_xlabel("Predicted Probability (Hazardous)")
        ax.set_ylabel("Count")
        ax.set_title("Probability Distribution")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Classification Report
    st.markdown("#### Classification Report")
    st.code(metrics["classification_report"], language=None)


# ─────────────────────────────────────────────
# PAGE: RISK PREDICTION
# ─────────────────────────────────────────────
elif page == "🔭 Risk Prediction" and data_loaded:
    st.markdown("# 🔭 Asteroid Risk Prediction")
    st.markdown(f"<span style='color:{MUTED}'>Enter asteroid parameters to classify its hazard level</span>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🛸 Input Parameters")

        abs_mag = st.slider(
            "Absolute Magnitude (H)", min_value=10.0, max_value=35.0, value=22.0, step=0.1,
            help="Lower = larger/brighter asteroid. Hazardous: typically < 22"
        )
        diameter = st.slider(
            "Estimated Diameter (km)", min_value=0.001, max_value=5.0, value=0.1, step=0.001, format="%.3f",
            help="Mean diameter in kilometers. Hazardous: > 0.14 km"
        )
        velocity = st.slider(
            "Relative Velocity (km/s)", min_value=0.5, max_value=80.0, value=15.0, step=0.5,
            help="Speed relative to Earth at closest approach"
        )
        miss_dist = st.slider(
            "Miss Distance (million km)", min_value=0.01, max_value=75.0, value=5.0, step=0.1,
            help="Closest approach distance. Hazardous: < 7.5 million km"
        )
        eccentricity = st.slider(
            "Orbital Eccentricity", min_value=0.0, max_value=1.0, value=0.4, step=0.01,
            help="Shape of orbit. 0 = circular, 1 = parabolic. Higher = more eccentric"
        )

        predict_btn = st.button("🔭 Analyze Asteroid", use_container_width=True)

    with col2:
        st.markdown("### 📡 Analysis Result")

        if predict_btn:
            input_data = {
                "absolute_magnitude_h": abs_mag,
                "diameter_min_km": diameter * 0.8,
                "diameter_max_km": diameter * 1.2,
                "diameter_mean_km": diameter,
                "relative_velocity_km_s": velocity,
                "miss_distance_km": miss_dist * 1_000_000,
                "orbital_eccentricity": eccentricity,
            }

            result = predict_single(model, input_data, feature_names)
            prob_haz = result["probability_hazardous"]
            prob_safe = result["probability_safe"]
            label = result["label"]

            if result["prediction"] == 1:
                st.markdown(f"""
                <div class="hazard-box">
                    <h2 style="color:#ff4444; margin:0">⚠️ POTENTIALLY HAZARDOUS</h2>
                    <h3 style="color:#ff6b6b; margin:8px 0">Probability: {prob_haz:.1%}</h3>
                    <p style="color:#ffaaaa; margin:0">This asteroid meets the criteria for potential Earth impact risk</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-box">
                    <h2 style="color:#00ff7f; margin:0">✅ SAFE</h2>
                    <h3 style="color:#00cc66; margin:8px 0">Confidence: {prob_safe:.1%}</h3>
                    <p style="color:#aaffcc; margin:0">This asteroid does not pose a significant hazard to Earth</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Probability Breakdown")
            fig, ax = plt.subplots(figsize=(5, 2))
            bar_colors = ["#00ff7f", "#ff4444"]
            bars = ax.barh(["Safe", "Hazardous"], [prob_safe, prob_haz], color=bar_colors)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="white", linestyle="--", linewidth=1, alpha=0.5)
            for bar, prob in zip(bars, [prob_safe, prob_haz]):
                ax.text(min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                       f"{prob:.1%}", va="center", color=TEXT, fontsize=11, fontweight="bold")
            ax.set_xlabel("Probability")
            ax.grid(axis="x", alpha=0.3)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        else:
            st.info("👈 Adjust the parameters and click **Analyze Asteroid** to get a classification.")
            st.markdown(f"""
            <div style='background:{CARD_BG}; border:1px solid #30363d; border-radius:10px; padding:16px; color:{MUTED}'>
            <h4 style='color:{ACCENT}'>📖 How to Use</h4>
            <ul>
                <li>Set the asteroid's <b>absolute magnitude</b> — lower means bigger</li>
                <li>Estimate its <b>diameter</b> and <b>velocity</b></li>
                <li>Enter the <b>miss distance</b> — how close it passes Earth</li>
                <li>Set the <b>orbital eccentricity</b> — how elliptical the orbit is</li>
                <li>Click Analyze to get the hazard classification</li>
            </ul>
            <hr style='border-color:#30363d'>
            <small>An asteroid is classified as potentially hazardous (PHA) if it passes within 0.05 AU (~7.5M km) 
            and has an absolute magnitude ≤ 22 (diameter ≥ ~140m).</small>
            </div>
            """, unsafe_allow_html=True)

elif not data_loaded:
    st.warning("⚠️ Could not load data. Check your NASA API key in `src/data_loader.py`.")
