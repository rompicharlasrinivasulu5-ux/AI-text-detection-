import joblib
from pipeline import clean_text, extract_stylometric_features

# =====================================
# LOAD MODEL
# =====================================
model_pipeline = joblib.load("model_pipeline.joblib")

# =====================================
# PREDICT FUNCTION
# =====================================
def predict(text):

    # Clean text
    text_clean = clean_text(text)

    # Extract features
    stylo = extract_stylometric_features(text_clean).reshape(1, -1)

    # Prediction
    prediction = model_pipeline.predict(stylo)[0]

    # Raw probabilities
    probs = model_pipeline.predict_proba(stylo)[0]

    # Original probabilities
    human_prob_raw = probs[0]
    ai_prob_raw = probs[1]

    # Confidence smoothing
    if prediction == 0:

        label = "Human-written"

        confidence = 0.75 + (human_prob_raw * 0.20)

    else:

        label = "AI-generated"

        confidence = 0.75 + (ai_prob_raw * 0.20)

    return (
        label,
        confidence,
        human_prob_raw,
        ai_prob_raw
    )
