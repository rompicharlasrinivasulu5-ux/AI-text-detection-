import streamlit as st
import plotly.graph_objects as go
from predict import predict
import time

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="AI Text Detector PRO",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================
# CUSTOM CSS
# =====================================
st.markdown("""
<style>

/* Main App Background */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

/* Remove Streamlit Header/Footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Title */
.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: white;
    margin-top: 10px;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #f1f5f9;
    margin-bottom: 30px;
}

/* Glassmorphism Card */
.glass {
    background: rgba(255, 255, 255, 0.15);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.2);
}

/* Result Box */
.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: white;
    margin-top: 20px;
}

.human {
    background: linear-gradient(135deg, #22c55e, #16a34a);
}

.ai {
    background: linear-gradient(135deg, #ef4444, #dc2626);
}

/* Footer */
.footer {
    text-align: center;
    font-size: 16px;
    color: white;
    margin-top: 50px;
    padding-bottom: 20px;
}

/* Text Area */
textarea {
    font-size: 16px !important;
}

/* Button */
.stButton button {
    background: linear-gradient(135deg, #38bdf8, #2563eb);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    border: none;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================
st.markdown(
    "<div class='main-title'>🤖 AI Text Detector PRO</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Detect whether a text is Human-written or AI-generated using Machine Learning & NLP</div>",
    unsafe_allow_html=True
)

# =====================================
# LAYOUT
# =====================================
col1, col2 = st.columns([2, 1])

# =====================================
# LEFT PANEL
# =====================================
with col1:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    text = st.text_area(
        "✍️ Enter your text (Minimum 100 words):",
        height=300,
        placeholder="Paste or type your text here..."
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================
# RIGHT PANEL
# =====================================
with col2:

    st.markdown("""
    <div class='glass'>
    <h3>💡 Tips</h3>

    ✅ Use longer text<br>
    ✅ Try both AI & human text<br>
    ✅ Observe confidence levels<br>
    ✅ Minimum 100 words recommended
    </div>
    """, unsafe_allow_html=True)

# =====================================
# ANALYZE BUTTON
# =====================================
if st.button("🚀 Analyze Text"):

    if len(text.split()) < 100:
        st.warning("⚠️ Please enter at least 100 words.")
        st.stop()

    # Loading animation
    with st.spinner("Analyzing text..."):

        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

    # Prediction
    label, prob = predict(text)

    st.markdown("---")

    # =====================================
    # RESULT DISPLAY
    # =====================================
    if label == "Human-written":

        st.markdown(
            "<div class='result-box human'>🧑 HUMAN-WRITTEN</div>",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            "<div class='result-box ai'>🤖 AI-GENERATED</div>",
            unsafe_allow_html=True
        )

    # =====================================
    # CONFIDENCE LEVEL
    # =====================================
    st.markdown("## 📈 Confidence Gauge")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': "%"},
        title={'text': "Model Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ffffff"},
            'steps': [
                {'range': [0, 50], 'color': "#ef4444"},
                {'range': [50, 75], 'color': "#facc15"},
                {'range': [75, 100], 'color': "#22c55e"}
            ]
        }
    ))

    gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'size': 18}
    )

    st.plotly_chart(gauge, use_container_width=True)

    # =====================================
    # PROBABILITY DISTRIBUTION
    # =====================================
    st.markdown("## 📊 Probability Distribution")

    bar = go.Figure(data=[
        go.Bar(
            x=["Human", "AI"],
            y=[1 - prob, prob],
            text=[
                f"{(1 - prob)*100:.1f}%",
                f"{prob*100:.1f}%"
            ],
            textposition='auto',
            marker_color=["#22c55e", "#ef4444"]
        )
    ])

    bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        yaxis=dict(range=[0,1])
    )

    st.plotly_chart(bar, use_container_width=True)

    # =====================================
    # INTERPRETATION BOX
    # =====================================
    st.markdown(f"""
    <div class='glass'>
    <h3>🧠 Interpretation</h3>

    The model predicts this text as:
    <b>{label}</b>

    with a confidence score of:
    <b>{prob:.2f}</b>

    The prediction is based on:
    <ul>
        <li>Writing style patterns</li>
        <li>Sentence structure</li>
        <li>Word distribution</li>
        <li>Linguistic behavior</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================
# FOOTER
# =====================================
st.markdown("""
<div class='footer'>
✨ Developed by <b>Srinivasulu Rompicharla</b><br>
AI Text Detection System using Machine Learning & NLP
</div>
""", unsafe_allow_html=True)
