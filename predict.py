import joblib
import numpy as np

from pipeline import clean_text, extract_stylometric_features

# Load FULL pipeline (scaler + model)
pipe = joblib.load("bkp/model_pipeline.joblib")


def predict(text):
    # Step 0: Length check
    if len(text.split()) < 100:
        return "Input too short (minimum 100 words required)", 0.0

    # Step 1: Clean
    text_clean = clean_text(text)

    # Step 2: Features
    features = extract_stylometric_features(text_clean).reshape(1, -1)

    # Step 3: Predict (pipeline handles scaling)
    probability = pipe.predict_proba(features)[0][1]

    # Step 4: Label (standard threshold)
    if probability > 0.58:
        label = "AI-generated"
    else:
        label = "Human-written"

    return label, probability


# Test
if __name__ == "__main__":
    text = """In today’s digital era, technology plays a crucial role in shaping the way individuals interact, communicate, and perform daily tasks. The advancement of digital tools has made it possible to access information quickly and efficiently, thereby improving productivity and convenience. Furthermore, the widespread adoption of automation technologies has reduced the need for manual intervention in many processes. As a result, organizations are able to operate more efficiently and achieve better outcomes. However, it is also important to consider the potential challenges associated with over-reliance on technology, including security risks and reduced human involvement in critical decision-making processes. Through the application of machine learning algorithms, organizations are able to identify patterns, automate processes, and enhance decision-making capabilities."""

    label, prob = predict(text)

    print("Prediction:", label)
    print("Confidence:", prob)
