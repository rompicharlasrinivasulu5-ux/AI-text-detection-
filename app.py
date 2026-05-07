import streamlit as st
import plotly.graph_objects as go
from predict import predict
import time

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="AI Text Detector Pro",
    page_icon="🤖",
    layout="wide"
)

# ==============================
# BEAUTIFUL BACKGROUND + GLASS UI
# ==============================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}
.glass {
    background: rgba(255,255,255,0.15);
    border-radius: 15px;
    padding: 20px;
    backdrop-filter: blur(15px);
    margin-bottom: 20px;
}
.result {
    font-size: 26px;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border-radius: 10px;
}
.human {
    background: linear-gradient(135deg, #22c55e, #16a34a);
}
.ai {
    background: linear-gradient(135deg, #ef4444, #dc2626);
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown("<div class='title'>🤖 AI Text Detector PRO</div>",
            unsafe_allow_html=True)

st.markdown("""
<div class='glass'>
🔍 Detect whether text is Human-written or AI-generated using Machine Learning.
</div>
""", unsafe_allow_html=True)

# ==============================
# INPUT
# ==============================
col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area("✍️ Enter your text (100+ words):", height=250)

with col2:
    st.markdown("""
<div class='glass'>
💡 Tips
- Use longer text  
- Try both AI & human styles  
- Watch confidence graph  
</div>
""", unsafe_allow_html=True)

# ==============================
# BUTTON
# ==============================
if st.button("🚀 Analyze Text", use_container_width=True):

    if len(text.split()) < 100:
        st.warning("⚠️ Please enter at least 100 words.")
        st.stop()

    with st.spinner("Analyzing..."):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

    label, prob = predict(text)

    st.markdown("---")

    # RESULT
    if label == "Human-written":
        st.markdown("<div class='result human'>🧑 HUMAN-WRITTEN</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='result ai'>🤖 AI-GENERATED</div>",
                    unsafe_allow_html=True)

    # ==============================
    # GAUGE
    # ==============================
    st.subheader("📈 Confidence Gauge")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Confidence %"},
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

    st.plotly_chart(gauge, use_container_width=True)

    # ==============================
    # BAR CHART
    # ==============================
    st.subheader("📊 Probability Distribution")

    bar = go.Figure(data=[
        go.Bar(
            x=["Human", "AI"],
            y=[1 - prob, prob],
            marker_color=["#22c55e", "#ef4444"]
        )
    ])

    st.plotly_chart(bar, use_container_width=True)

    # ==============================
    # INFO BOX
    # ==============================
    st.markdown("""
<div class='glass'>
🧠 The model analyzes writing style, sentence structure, and linguistic patterns.
</div>
""", unsafe_allow_html=True)

# ==============================
# FOOTER (YOUR NAME)
# ==============================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:white; font-size:16px;'>
✨ Developed by <b>Srinivasulu Rompicharla</b>
</div>
""", unsafe_allow_html=True)
