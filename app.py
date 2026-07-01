import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("iris_model.pkl")
    scaler = joblib.load("iris_scaler.pkl")
    le = joblib.load("iris_label_encoder.pkl")
    metrics = joblib.load("iris_metrics.pkl")
    return model, scaler, le, metrics

@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv").drop(columns=["Id"])
    return df

model, scaler, le, metrics = load_artifacts()
df = load_data()

FEATURES = metrics["feature_names"]
CLASSES = metrics["classes"]

SPECIES_INFO = {
    "Iris-setosa": {
        "emoji": "🌱",
        "color": "#e75480",
        "desc": "Smallest of the three species, with distinctly short and wide petals. Easiest to tell apart.",
    },
    "Iris-versicolor": {
        "emoji": "🌷",
        "color": "#9b59b6",
        "desc": "Medium-sized, with measurements that often overlap with Virginica — the trickiest to classify.",
    },
    "Iris-virginica": {
        "emoji": "🌺",
        "color": "#c0392b",
        "desc": "Largest of the three species, with the longest petals and sepals.",
    },
}


st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #e75480, #9b59b6, #c0392b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #fafafa;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #eee;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌸 Iris Flower Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict the species of an Iris flower from its sepal & petal measurements</p>', unsafe_allow_html=True)


st.sidebar.header("🔧 Flower Measurements")
st.sidebar.caption("Adjust the sliders to match your flower")

sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df.SepalLengthCm.min()), float(df.SepalLengthCm.max()), 5.8, 0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df.SepalWidthCm.min()), float(df.SepalWidthCm.max()), 3.0, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", float(df.PetalLengthCm.min()), float(df.PetalLengthCm.max()), 3.7, 0.1)
petal_width = st.sidebar.slider("Petal Width (cm)", float(df.PetalWidthCm.min()), float(df.PetalWidthCm.max()), 1.2, 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("🎲 Or try a random sample")
if st.sidebar.button("Pick random flower from dataset"):
    sample = df.sample(1).iloc[0]
    sepal_length, sepal_width = sample.SepalLengthCm, sample.SepalWidthCm
    petal_length, petal_width = sample.PetalLengthCm, sample.PetalWidthCm
    st.sidebar.success(f"Loaded a real {sample.Species} sample!")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")
st.sidebar.write(f"**Best model:** {metrics['best_model_name']}")
st.sidebar.write(f"**Test Accuracy:** {metrics['all_results'][metrics['best_model_name']]['accuracy']*100:.1f}%")

# ---------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------
col1, col2 = st.columns([1, 1.2])

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_scaled = scaler.transform(input_data)
pred_encoded = model.predict(input_scaled)[0]
pred_species = le.inverse_transform([pred_encoded])[0]
pred_proba = model.predict_proba(input_scaled)[0]

with col1:
    st.subheader("Prediction")
    info = SPECIES_INFO[pred_species]
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {info['color']}22, {info['color']}11);
                border: 2px solid {info['color']};
                border-radius: 16px; padding: 1.5rem; text-align: center;">
        <div style="font-size: 3rem;">{info['emoji']}</div>
        <div style="font-size: 1.6rem; font-weight: 700; color: {info['color']};">{pred_species}</div>
        <div style="color: #666; margin-top: 0.5rem;">{info['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("###  Prediction Confidence")
    proba_df = pd.DataFrame({
        "Species": CLASSES,
        "Probability": pred_proba
    }).sort_values("Probability", ascending=True)

    fig_proba = go.Figure(go.Bar(
        x=proba_df["Probability"],
        y=proba_df["Species"],
        orientation="h",
        marker_color=[SPECIES_INFO[s]["color"] for s in proba_df["Species"]],
        text=[f"{p*100:.1f}%" for p in proba_df["Probability"]],
        textposition="outside",
    ))
    fig_proba.update_layout(
        xaxis_range=[0, 1.1], height=250,
        margin=dict(l=0, r=30, t=10, b=10),
        xaxis_title="Probability", yaxis_title=""
    )
    st.plotly_chart(fig_proba, use_container_width=True)

    st.markdown("### 📏 Your Input")
    input_df = pd.DataFrame({
        "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
        "Value (cm)": [sepal_length, sepal_width, petal_length, petal_width]
    })
    st.dataframe(input_df, hide_index=True, use_container_width=True)

with col2:
    st.subheader("Where your flower sits")
    fig_scatter = px.scatter(
        df, x="PetalLengthCm", y="PetalWidthCm", color="Species",
        color_discrete_map={s: SPECIES_INFO[s]["color"] for s in CLASSES},
        opacity=0.6,
        labels={"PetalLengthCm": "Petal Length (cm)", "PetalWidthCm": "Petal Width (cm)"},
    )
    fig_scatter.add_trace(go.Scatter(
        x=[petal_length], y=[petal_width],
        mode="markers",
        marker=dict(size=20, color="black", symbol="star", line=dict(width=2, color="white")),
        name="Your Flower"
    ))
    fig_scatter.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("📈 Model Performance Comparison")
    perf_rows = []
    for name, r in metrics["all_results"].items():
        perf_rows.append({
            "Model": name,
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1 Score": r["f1"],
        })
    perf_df = pd.DataFrame(perf_rows)
    st.dataframe(
        perf_df.style.format({c: "{:.2%}" for c in ["Accuracy", "Precision", "Recall", "F1 Score"]})
                     .highlight_max(subset=["Accuracy", "Precision", "Recall", "F1 Score"], color="#f8d7e0"),
        hide_index=True, use_container_width=True
    )

st.markdown("---")

# ---------------------------------------------------------------
# EXPLORE DATASET SECTION
# ---------------------------------------------------------------
with st.expander("Explore the full dataset"):
    tab1, tab2, tab3 = st.tabs(["Raw Data", "Pairwise Plot", "Distributions"])

    with tab1:
        st.dataframe(df, use_container_width=True, height=300)
        st.caption(f"{len(df)} total samples, {len(CLASSES)} species, {len(FEATURES)} features")

    with tab2:
        selected_features = st.multiselect(
            "Choose features to compare", FEATURES, default=["PetalLengthCm", "PetalWidthCm"]
        )
        if len(selected_features) == 2:
            fig = px.scatter(
                df, x=selected_features[0], y=selected_features[1], color="Species",
                color_discrete_map={s: SPECIES_INFO[s]["color"] for s in CLASSES},
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pick exactly 2 features to see a scatter plot.")

    with tab3:
        feature_for_hist = st.selectbox("Feature", FEATURES)
        fig_hist = px.histogram(
            df, x=feature_for_hist, color="Species", marginal="box",
            color_discrete_map={s: SPECIES_INFO[s]["color"] for s in CLASSES},
            opacity=0.7, barmode="overlay"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

st.markdown(
    '<p style="text-align:center; color:#aaa; margin-top:2rem;">Built with scikit-learn + Streamlit 🌸</p>',
    unsafe_allow_html=True
)
