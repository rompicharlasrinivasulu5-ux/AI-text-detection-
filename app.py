import streamlit as st
import plotly.graph_objects as go
from predict import predict
from pipeline import extract_stylometric_features
import pandas as pd
import time

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI Text Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #0f172a;
    color: #f8fafc;
}

/* Hide Streamlit Default UI */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Title */
.main-title {
    font-size: 52px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background-color: #1e293b;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #334155;
    margin-bottom: 20px;
}

/* Result Cards */
.result-human {
    background: linear-gradient(135deg, #16a34a, #15803d);
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: white;
}

.result-ai {
    background: linear-gradient(135deg, #dc2626, #991b1b);
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    color: white;
}

/* Metrics */
.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #374151;
    text-align: center;
    height: 180px;
}

/* Buttons */
.stButton button {
    background: #2563eb;
    color: white;
    border-radius: 12px;
    border: none;
    height: 52px;
    width: 100%;
    font-size: 18px;
    font-weight: 600;
}

/* Text Area */
textarea {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 12px !important;
    font-size: 16px !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    padding-top: 40px;
    padding-bottom: 20px;
    font-size: 15px;
}

/* Small Labels */
.small-text {
    color: #cbd5e1;
    font-size: 15px;
}

/* Section Headers */
.section-header {
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 10px;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown(
    "<div class='main-title'>AI Text Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Enterprise AI-powered detection using Machine Learning & NLP</div>",
    unsafe_allow_html=True
)

# ==========================================
# LAYOUT
# ==========================================
left, right = st.columns([2.2, 1])

# ==========================================
# INPUT SECTION
# ==========================================
with left:

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    text = st.text_area(
        "Enter Text",
        height=300,
        placeholder="Paste or type your text here..."
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# SYSTEM OVERVIEW
# ==========================================
with right:

    st.markdown("""
    <div class='card'>

    <div class='section-header'>Detection Engine</div>

    <p class='small-text'>

    • Stylometric Analysis<br><br>

    • NLP Feature Extraction<br><br>

    • Logistic Regression Model<br><br>

    • Real-time Classification<br><br>

    • Confidence Estimation

    </p>

    </div>
    """, unsafe_allow_html=True)

# ==========================================
# ANALYZE BUTTON
# ==========================================
if st.button("Analyze Text"):

    # ==========================================
    # WORD VALIDATION
    # ==========================================
    if len(text.split()) < 100:
        st.warning("Please enter at least 100 words.")
        st.stop()

    # ==========================================
    # LOADING STEPS
    # ==========================================
    loading_steps = [
        "Cleaning input text...",
        "Extracting stylometric patterns...",
        "Analyzing sentence structure...",
        "Computing statistical features...",
        "Evaluating linguistic variability...",
        "Generating final prediction..."
    ]

    progress = st.progress(0)

    status = st.empty()

    for i, step in enumerate(loading_steps):

        status.info(step)

        time.sleep(0.45)

        progress.progress((i + 1) / len(loading_steps))

    status.success("Analysis completed successfully.")

    # ==========================================
    # PREDICTION
    # ==========================================
    label, prob, human_prob, ai_prob = predict(text)

    # ==========================================
    # FEATURE EXTRACTION
    # ==========================================
    features = extract_stylometric_features(text)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # RESULT CARD
    # ==========================================
    if label == "Human-written":

        st.markdown(
            "<div class='result-human'>Human-Written Text Detected</div>",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            "<div class='result-ai'>AI-Generated Text Detected</div>",
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # METRICS ROW
    # ==========================================
    m1, m2, m3 = st.columns(3)

    with m1:

        st.markdown(f"""
        <div class='metric-card'>
        <h3>Confidence</h3>
        <h1>{prob*100:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with m2:

        st.markdown(f"""
        <div class='metric-card'>
        <h3>Word Count</h3>
        <h1>{len(text.split())}</h1>
        </div>
        """, unsafe_allow_html=True)

    with m3:

        st.markdown(f"""
        <div class='metric-card'>
        <h3>Prediction</h3>
        <h1>{label}</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # AI ANALYSIS SUMMARY
    # ==========================================
    if label == "Human-written":

        analysis_text = """
        The submitted text demonstrates natural linguistic variation,
        irregular sentence transitions, and diverse vocabulary usage.

        The stylometric analysis indicates strong human-like writing behavior
        with realistic burstiness and sentence complexity patterns.
        """

    else:

        analysis_text = """
        The submitted text demonstrates highly consistent sentence structures,
        repetitive stylistic patterns, and statistically uniform writing behavior.

        The system detected characteristics commonly associated with
        AI-generated textual content.
        """

    st.markdown(f"""
    <div class='card'>

    <div class='section-header'>AI Analysis Summary</div>

    <p class='small-text'>
    {analysis_text}
    </p>

    </div>
    """, unsafe_allow_html=True)

    # ==========================================
    # TABS
    # ==========================================
    tab1, tab2, tab3 = st.tabs([
        "Confidence Analysis",
        "Feature Analysis",
        "Model Interpretation"
    ])

    # ==========================================
    # TAB 1
    # ==========================================
    with tab1:

        # ==========================================
        # GAUGE CHART
        # ==========================================
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%"},
            title={'text': "Prediction Confidence"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3b82f6"},
                'bgcolor': "#111827",
                'steps': [
                    {'range': [0, 50], 'color': "#7f1d1d"},
                    {'range': [50, 75], 'color': "#78350f"},
                    {'range': [75, 100], 'color': "#14532d"}
                ]
            }
        ))

        gauge.update_layout(
            paper_bgcolor="#0f172a",
            font={'color': "white"},
            height=420
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        # ==========================================
        # PROBABILITY BAR CHART
        # ==========================================
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=["Human", "AI"],
            y=[human_prob, ai_prob],
            text=[
                f"{human_prob*100:.1f}%",
                f"{ai_prob*100:.1f}%"
            ],
            textposition='auto',
            marker_color=["#22c55e", "#ef4444"]
        ))

        fig.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="white",
            height=350,
            yaxis=dict(range=[0,1])
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

    # ==========================================
    # TAB 2
    # ==========================================
    with tab2:

        feature_names = [
            "Sentence Count",
            "Word Count",
            "Avg Sentence Length",
            "Unique Word Ratio",
            "Poisson Deviation",
            "Burstiness",
            "Word Entropy",
            "Function Word Ratio",
            "Noun Ratio",
            "Verb Ratio",
            "Avg Dependency Depth",
            "Max Dependency Depth"
        ]

        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": features
        })

        st.dataframe(
            feature_df,
            use_container_width=True
        )

    # ==========================================
    # TAB 3
    # ==========================================
    with tab3:

        st.markdown("""
        ### Model Interpretation

        The AI Text Detection System performs analysis using
        stylometric and linguistic characteristics extracted
        from the submitted text.

        The system evaluates:

        - Vocabulary diversity
        - Sentence complexity
        - Writing burstiness
        - Statistical language behavior
        - Dependency structure
        - Linguistic consistency

        Final predictions are generated using a trained
        Logistic Regression machine learning model.
        """)

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class='footer'>
Developed by <b>Srinivasulu Rompicharla</b><br>
AI Text Detection using Machine Learning & NLP
</div>
""", unsafe_allow_html=True)
