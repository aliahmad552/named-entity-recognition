import tensorflow as tf
import re
import numpy as np
import string
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
MAX_LEN = 75
#loading

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

with open(r'D:\AI_ML_Projects\named_entity_recognition_bilstm_crf\app\word2index (4).pkl', "rb") as f:
    word2index = pickle.load(f)

model = load_model(r'D:\AI_ML_Projects\named_entity_recognition_bilstm_crf\app\ner_model.h5')

word2index["--UNKNOWN_WORD--"]=0

word2index["--PADDING--"]=1

re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")
sentence_words = re_tok.sub(r" \1 ", "Ali ahmad lives in Lahore and working in devsinc company.").split()

X = [word2index.get(w, word2index["--UNKNOWN_WORD--"]) for w in sentence_words]
X = pad_sequences(maxlen=MAX_LEN, sequences=[X], padding="post", value=word2index["--PADDING--"])

pred = model.predict(X)
pred = np.argmax(pred, axis=-1)[0]
retval = ""
for w, p in zip(sentence_words, pred[:len(sentence_words)]):  # ignore padding predictions
    retval += f"{w:15}: {index2tag[p]}\n"

print(retval)