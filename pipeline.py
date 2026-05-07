# ===============================
# IMPORTS
# ===============================
import re
import numpy as np
import math
from collections import Counter
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

import spacy

#from transformers import BertTokenizer, BertModel
#import torch

# ===============================
# DOWNLOAD NLTK RESOURCES
# ===============================
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ===============================
# LOAD MODELS (ONLY ONCE)
# ===============================
# spaCy
nlp = spacy.load("en_core_web_sm")

# Stopwords
stop_words = set(stopwords.words('english'))

# BERT
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#bert_model = BertModel.from_pretrained('bert-base-uncased')
#bert_model.eval()

# ===============================
# GLOBAL WORD PROBABILITY
# ===============================
df_temp = pd.read_csv("data/processed_data.csv")

all_words = []

for text in df_temp['clean_text']:
    tokens = word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha()]
    all_words.extend(tokens)

word_freq = Counter(all_words)

total_words = sum(word_freq.values())
word_prob = {word: count / total_words for word, count in word_freq.items()}

print("Word probability dictionary created!")

# ===============================
# CLEAN TEXT
# ===============================


def clean_text(text):
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,!?\' ]', '', text)
    return text.strip()

# ===============================
# STYLOMETRIC FEATURES
# ===============================


def extract_stylometric_features(text):
    tokens = word_tokenize(text.lower())
    tokens_alpha = [t for t in tokens if t.isalpha()]

    # Sentence count
    sentences = sent_tokenize(text)
    sentence_count = len(sentences)

    # Word count
    word_count = len(tokens)

    # Avg sentence length
    avg_sentence_length = word_count / max(sentence_count, 1)

    # Unique word ratio
    unique_word_ratio = len(set(tokens)) / max(len(tokens), 1)

    # Poisson deviation
    deviation = 0
    for word in tokens_alpha:
        expected = word_prob.get(word, 1e-6)
        deviation += abs(1 - expected)
    poisson_deviation = deviation / max(len(tokens_alpha), 1)

    # Burstiness
    lengths = [len(word_tokenize(s)) for s in sentences]
    if len(lengths) <= 1:
        burstiness = 0
    else:
        burstiness = np.std(lengths) / (np.mean(lengths) + 1e-6)

    # Entropy
    if len(tokens_alpha) == 0:
        word_entropy = 0
    else:
        counts = Counter(tokens_alpha)
        probs = [c / len(tokens_alpha) for c in counts.values()]
        word_entropy = -sum(p * math.log(p) for p in probs)

    # Function word ratio
    func_count = sum(1 for t in tokens if t in stop_words)
    function_word_ratio = func_count / max(len(tokens), 1)

    # POS + Dependency
    doc = nlp(text)

    pos_counts = {}
    depths = []

    for token in doc:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

        depth = 0
        head = token
        while head.head != head:
            depth += 1
            head = head.head
        depths.append(depth)

    total_tokens = len(doc)

    noun_ratio = pos_counts.get('NOUN', 0) / max(total_tokens, 1)
    verb_ratio = pos_counts.get('VERB', 0) / max(total_tokens, 1)

    avg_dep_depth = sum(depths) / max(len(depths), 1)
    max_dep_depth = max(depths) if depths else 0

    return np.array([
        sentence_count,
        word_count,
        avg_sentence_length,
        unique_word_ratio,
        poisson_deviation,
        burstiness,
        word_entropy,
        function_word_ratio,
        noun_ratio,
        verb_ratio,
        avg_dep_depth,
        max_dep_depth
    ])

# # ===============================
# # BERT EMBEDDING
# # ===============================


#def get_bert_embedding(text):
#    inputs = tokenizer(
#        text,
#        return_tensors='pt',
#        truncation=True,
#        padding=True,
#        max_length=512
#    )
#
#    with torch.no_grad():
#        outputs = bert_model(**inputs)
#
#   cls_embedding = outputs.last_hidden_state[:, 0, :]
#return cls_embedding.squeeze().numpy()
