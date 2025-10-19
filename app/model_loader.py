import os
import re
import string
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- CONFIG: update if your folders are different ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../model/ner_model.h5")      # your model path
WORD2INDEX_PATH = os.path.join(BASE_DIR, "../model/word2index (4).pkl")
MAX_LEN = 75   # same as notebook

# --- Load model & mappings once on import ---
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded from:", MODEL_PATH)
except Exception as e:
    model = None
    print("❌ Failed to load model:", MODEL_PATH)
    print(e)

with open(WORD2INDEX_PATH, "rb") as f:
    word2index = pickle.load(f)

index2tag = {1: 'B-gpe',
 2: 'I-tim',
 3: 'I-nat',
 4: 'B-per',
 5: 'B-eve',
 6: 'I-gpe',
 7: 'I-per',
 8: 'I-geo',
 9: 'B-org',
 10: 'B-tim',
 11: 'B-nat',
 12: 'I-eve',
 13: 'I-org',
 14: 'O',
 15: 'B-geo',
 16: 'I-art',
 17: 'B-art',
 0: '--PADDING--'}

# Ensure special tokens exist (make safe)
if "--UNKNOWN_WORD--" not in word2index:
    # append at end if missing
    word2index["--UNKNOWN_WORD--"] = max(word2index.values()) + 1
if "--PADDING--" not in word2index:
    word2index["--PADDING--"] = max(word2index.values()) + 1

# regular expression tokenization exactly like your notebook
RE_TOK = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")

def debug_info(sentence_words, indices, pred_indices=None, top_probs=None):
    print("---- DEBUG ----")
    print("Tokens:", sentence_words)
    print("Indices:", indices)
    if pred_indices is not None:
        print("Pred indices:", pred_indices[:len(sentence_words)])
    if top_probs is not None:
        print("Top probs (first token):", top_probs[0] if len(top_probs)>0 else None)
    print("---------------")

def predict_sentence(sentence, model, word2index, index2tag):
    # Step 1: Split sentence into words (same as training)
    sentence_words = sentence.split()

    # Step 2: Convert words to indices
    X = [[word2index.get(w, word2index.get("--UNKNOWN_WORD--", 1)) for w in sentence_words]]

    # Step 3: Pad sequence
    X = pad_sequences(maxlen=MAX_LEN, sequences=X, padding="post", value=word2index["--PADDING--"])

    # Step 4: Predict using trained model
    pred = model.predict(X)
    pred = np.argmax(pred, axis=-1)[0]  # take first sentence predictions

    # Step 5: Decode tags
    entities = []
    for w, p in zip(sentence_words, pred[:len(sentence_words)]):
        entities.append({
            "word": w,
            "entity": index2tag[int(p)]
        })

    return entities