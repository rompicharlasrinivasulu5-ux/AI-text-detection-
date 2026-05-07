import joblib
from pipeline import clean_text, extract_stylometric_features

# =====================================
# LOAD PIPELINE MODEL
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

    # Predict
    prediction = model_pipeline.predict(stylo)[0]

    # Probabilities
    probs = model_pipeline.predict_proba(stylo)[0]

    # Human probability
    human_prob = probs[0]

    # AI probability
    ai_prob = probs[1]

    # Final label + confidence
    if prediction == 0:

        label = "Human-written"

        confidence = 0.75 + (human_prob * 0.20)
    else:
        label = "AI-generated"
        confidence = 0.75 + (ai_prob * 0.20)

    return label, confidence
