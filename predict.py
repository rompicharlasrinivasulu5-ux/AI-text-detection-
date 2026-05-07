import joblib
from pipeline import clean_text, extract_stylometric_features

# =====================================
# LOAD MODEL PIPELINE
# =====================================
model_pipeline = joblib.load("model_pipeline.joblib")


# =====================================
# PREDICT FUNCTION
# =====================================
def predict(text):

    # =====================================
    # CLEAN TEXT
    # =====================================
    text_clean = clean_text(text)

    # =====================================
    # EXTRACT STYLOMETRIC FEATURES
    # =====================================
    stylo = extract_stylometric_features(text_clean).reshape(1, -1)

    # =====================================
    # MODEL PREDICTION
    # =====================================
    prediction = model_pipeline.predict(stylo)[0]

    # =====================================
    # RAW PROBABILITIES
    # =====================================
    probs = model_pipeline.predict_proba(stylo)[0]

    # Raw class probabilities
    human_prob_raw = probs[0]
    ai_prob_raw = probs[1]

    # =====================================
    # CONFIDENCE SMOOTHING
    # =====================================
    # Makes UI look more realistic
    # instead of extreme 0% / 100%

    human_prob = 0.75 + (human_prob_raw * 0.20)
    ai_prob = 0.75 + (ai_prob_raw * 0.20)

    # =====================================
    # NORMALIZE PROBABILITIES
    # =====================================
    total = human_prob + ai_prob

    human_prob = human_prob / total
    ai_prob = ai_prob / total

    # =====================================
    # FINAL LABEL + CONFIDENCE
    # =====================================
    if prediction == 0:

        label = "Human-written"

        confidence = human_prob

    else:

        label = "AI-generated"

        confidence = ai_prob

    # =====================================
    # RETURN RESULTS
    # =====================================
    return (
        label,
        confidence,
        human_prob,
        ai_prob
    )
