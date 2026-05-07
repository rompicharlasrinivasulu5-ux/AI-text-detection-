from pipeline import *

text = "This is a human written sentence."

cleaned = clean_text(text)
stylo = extract_stylometric_features(cleaned)
bert = get_bert_embedding(cleaned)

print(stylo.shape)   # (12,)
print(bert.shape)    # (768,)
