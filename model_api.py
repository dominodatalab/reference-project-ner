import pickle
import re
import string

import numpy as np

from keras import layers

from keras.models import Model
from keras.models import Input

from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils

with open("index/word2index.pickle", "rb") as handle:
    word2index = pickle.load(handle)

with open("index/index2tag.pickle", "rb") as handle:
    index2tag = pickle.load(handle)
    
with open("model/model_settings.pickle", "rb") as handle:
    model_settings = pickle.load(handle)

WORD_COUNT = len(word2index)
DENSE_EMBEDDING = model_settings["DENSE_EMBEDDING"] 
LSTM_UNITS = model_settings["LSTM_UNITS"]
LSTM_DROPOUT = model_settings["LSTM_DROPOUT"]
DENSE_UNITS = model_settings["DENSE_UNITS"]
TAG_COUNT = model_settings["TAG_COUNT"]
MAX_SENTENCE = model_settings["MAX_SENTENCE"]

input_layer = layers.Input(shape=(MAX_SENTENCE,))
model = layers.Embedding(WORD_COUNT, DENSE_EMBEDDING, embeddings_initializer="uniform", 
                             input_length=MAX_SENTENCE)(input_layer)
model = layers.Bidirectional(layers.LSTM(LSTM_UNITS, recurrent_dropout=LSTM_DROPOUT, return_sequences=True))(model)
model = layers.TimeDistributed(layers.Dense(DENSE_UNITS, activation="relu"))(model)

crf_layer = CRF(units=TAG_COUNT)
output_layer = crf_layer(model)

ner_model = Model(input_layer, output_layer)

save_load_utils.load_all_weights(ner_model, "model/ner_model.h5")


def score(sentence):
    
    re_tok = re.compile(f"([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])")
    sentence = re_tok.sub(r' \1 ', sentence).split()
    
    padded_sentence = sentence + [word2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence))
    padded_sentence = [word2index.get(w, 0) for w in padded_sentence]
    
    pred = ner_model.predict(np.array([padded_sentence]))
    pred = np.argmax(pred, axis=-1)

    retval = ""
    for w, p in zip(sentence, pred[0]):
        retval = retval + "{:15}: {:5}".format(w, index2tag[p]) + "\n"

    return retval

if __name__=='__main__':
    
    test_sentence = "President Obama became the first sitting American president to visit Hiroshima"
    print(score(test_sentence))
