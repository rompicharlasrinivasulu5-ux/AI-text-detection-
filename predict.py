import joblib
from pipeline import clean_text, extract_stylometric_features

# =====================================
# LOAD MODEL
# =====================================
model = joblib.load("model.joblib")
scaler_s = joblib.load("scaler_s.joblib")

# =====================================
# PREDICT FUNCTION
# =====================================
def predict(text):

    # Clean text
    text_clean = clean_text(text)

    # Extract stylometric features
    stylo = extract_stylometric_features(text_clean).reshape(1, -1)

    # Scale features
    stylo_scaled = scaler_s.transform(stylo)

    # Prediction
    prediction = model.predict(stylo_scaled)[0]

    # Probabilities
    probs = model.predict_proba(stylo_scaled)[0]

    # Human probability
    human_prob = probs[0]

    # AI probability
    ai_prob = probs[1]

    # =====================================
    # LABEL + CONFIDENCE
    # =====================================
    if prediction == 0:

        label = "Human-written"

        confidence = human_prob

    else:

        label = "AI-generated"

        confidence = ai_prob

    return label, confidence
