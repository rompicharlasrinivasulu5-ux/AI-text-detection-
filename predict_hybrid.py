import joblib
import numpy as np

from pipeline import clean_text, extract_stylometric_features, get_bert_embedding

# Load hybrid model
data = joblib.load("hybrid_model.joblib")

scaler_s = data["scaler_s"]
scaler_t = data["scaler_t"]
model = data["model"]


def predict(text):
    if len(text.split()) < 100:
        return "Too short", 0.0

    text = clean_text(text)

    # Stylometric
    stylo = extract_stylometric_features(text).reshape(1, -1)
    stylo_scaled = scaler_s.transform(stylo)

    # BERT
    bert = get_bert_embedding(text).reshape(1, -1)
    bert_scaled = scaler_t.transform(bert)

    # Combine
    X = np.hstack((stylo_scaled, bert_scaled))

    prob = model.predict_proba(X)[0][1]

    if prob > 0.5:
        return "AI-generated", prob
    else:
        return "Human-written", prob


# Test
if __name__ == "__main__":
    text = """Honestly, yesterday was kind of all over the place. I went out to grab some groceries but ended up forgetting half the things I needed. I also ran into an old friend unexpectedly, which was nice but threw off my plans. We talked for a while, and it made me lose track of time more than I expected. When I got back home, I realized I didn’t feel like cooking at all, so I just ordered something quick instead. The rest of the evening was pretty relaxed, though I kept thinking about the work I had to finish the next day. I tried to focus on getting a few things done, but I was feeling a bit tired and distracted. Eventually, I just decided to take it easy and get some rest, hoping today would be a bit more productive."""
    label, prob = predict(text)

    print("Prediction:", label)
    print("Confidence:", prob)
