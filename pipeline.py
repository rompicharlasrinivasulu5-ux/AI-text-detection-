# ===============================
# IMPORTS
# ===============================
import re
import numpy as np
import math
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

import spacy

# ===============================
# DOWNLOAD NLTK RESOURCES
# ===============================
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ===============================
# LOAD MODELS
# ===============================
nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words('english'))

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

    # Average sentence length
    avg_sentence_length = word_count / max(sentence_count, 1)

    # Unique word ratio
    unique_word_ratio = len(set(tokens)) / max(len(tokens), 1)

    # ===============================
    # SIMPLIFIED POISSON DEVIATION
    # ===============================
    if len(tokens_alpha) == 0:
        poisson_deviation = 0
    else:
        counts = Counter(tokens_alpha)

        freqs = np.array(list(counts.values()))

        poisson_deviation = np.std(freqs)

    # ===============================
    # BURSTINESS
    # ===============================
    lengths = [len(word_tokenize(s)) for s in sentences]

    if len(lengths) <= 1:
        burstiness = 0
    else:
        burstiness = np.std(lengths) / (np.mean(lengths) + 1e-6)

    # ===============================
    # WORD ENTROPY
    # ===============================
    if len(tokens_alpha) == 0:
        word_entropy = 0
    else:
        counts = Counter(tokens_alpha)

        probs = [c / len(tokens_alpha) for c in counts.values()]

        word_entropy = -sum(p * math.log(p) for p in probs)

    # ===============================
    # FUNCTION WORD RATIO
    # ===============================
    func_count = sum(1 for t in tokens if t in stop_words)

    function_word_ratio = func_count / max(len(tokens), 1)

    # ===============================
    # POS + DEPENDENCY FEATURES
    # ===============================
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

    # ===============================
    # FINAL FEATURE VECTOR
    # ===============================
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
