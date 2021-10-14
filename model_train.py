#!/usr/local/anaconda/bin/python

import pickle
import sys
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_keras_history import plot_history
from sklearn.model_selection import train_test_split

from keras_contrib.utils import save_load_utils
from keras import layers 
from keras import optimizers
from keras.models import Model
from keras.models import Input
from keras_contrib.layers import CRF
from keras_contrib import losses
from keras_contrib import metrics

def load_data(filename="dataset/ner_dataset.csv", encoding="iso-8859-1", header=0):
    
    print("Loading data...", end =" ")
    data_df = pd.read_csv(filename, encoding=encoding, header=header)
    print("done.")
    
    return data_df

def to_tuples(data):
    iterator = zip(data["Word"].values.tolist(),
                   data["POS"].values.tolist(),
                   data["Tag"].values.tolist())
    
    return [(word, pos, tag) for word, pos, tag in iterator]

def pre_process(data_df):
    
    print("Pre-processing...")
    data_df = data_df.fillna(method="ffill")
    
    print("Total number of sentences in the dataset: {:,}".format(data_df["Sentence #"].nunique()))
    print("Total words in the dataset: {:,}".format(data_df.shape[0]))
    
    word_counts = data_df.groupby("Sentence #")["Word"].agg(["count"])
    MAX_SENTENCE = word_counts.max()[0]
    print("Longest sentence in the corpus contains {} words.".format(MAX_SENTENCE))
    
    all_words = list(set(data_df["Word"].values))
    all_tags = list(set(data_df["Tag"].values))
    
    print("Number of unique words: {}".format(data_df["Word"].nunique()))
    print("Number of unique tags : {}".format(data_df["Tag"].nunique()))
    
    # Generate word indicices   
    word2index = {word: idx + 2 for idx, word in enumerate(all_words)}
    word2index["--UNKNOWN_WORD--"]=0
    word2index["--PADDING--"]=1
    index2word = {idx: word for word, idx in word2index.items()}
    
    # Generate tag indicices
    tag2index = {tag: idx + 1 for idx, tag in enumerate(all_tags)}
    tag2index["--PADDING--"] = 0
    index2tag = {idx: word for word, idx in tag2index.items()}
    
    sentences = data_df.groupby("Sentence #").apply(to_tuples).tolist()
    
    print("Pre-processing completed.")
    
    return (word2index, index2word, tag2index, index2tag), sentences
    
def vectorize(word2index, index2word, tag2index, index2tag, sentences):
    
    print("Vectorizing...")
    
    MAX_SENTENCE = len(max([s for s in sentences], key=len))
    
    X = [[word[0] for word in sentence] for sentence in sentences]
    y = [[word[2] for word in sentence] for sentence in sentences]
    
    X = [[word2index[word] for word in sentence] for sentence in X]
    y = [[tag2index[tag] for tag in sentence] for sentence in y]
    
    # Apply padding
    X = [sentence + [word2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in X]
    y = [sentence + [tag2index["--PADDING--"]] * (MAX_SENTENCE - len(sentence)) for sentence in y]

    TAG_COUNT = len(tag2index)
    y = [ np.eye(TAG_COUNT)[sentence] for sentence in y]
    
    print("Vectorizing completed.")
    
    return X, y

def get_training_sets(X, y, test_size=0.1, random_state=1234):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print("Number of sentences in the training dataset: {}".format(len(X_train)))
    print("Number of sentences in the test dataset    : {}".format(len(X_test)))

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    return X_train, X_test, y_train, y_test

def get_model(X_train, y_train, WORD_COUNT, lr=0.001, DENSE_EMBEDDING = 50, LSTM_UNITS = 50, LSTM_DROPOUT = 0.1, DENSE_UNITS = 100):
    
    MAX_SENTENCE = X_train.shape[1]
    TAG_COUNT = y_train.shape[2]
    
    input_layer = layers.Input(shape=(MAX_SENTENCE,))
    model = layers.Embedding(WORD_COUNT, DENSE_EMBEDDING, embeddings_initializer="uniform", 
                             input_length=MAX_SENTENCE)(input_layer)
    model = layers.Bidirectional(layers.LSTM(LSTM_UNITS, recurrent_dropout=LSTM_DROPOUT, return_sequences=True))(model)
    model = layers.TimeDistributed(layers.Dense(DENSE_UNITS, activation="relu"))(model)

    crf_layer = CRF(units=TAG_COUNT)
    output_layer = crf_layer(model)

    ner_model = Model(input_layer, output_layer)

    loss = losses.crf_loss
    acc_metric = metrics.crf_accuracy

    opt = optimizers.Adam(lr=lr)

    ner_model.compile(optimizer=opt, loss=loss, metrics=[acc_metric])
    
    ner_model.summary()
    
    return ner_model

        
def save_model(ner_model, filename, word2index, index2tag):
    
    print("Persisting model...", end =" ")
    
    save_load_utils.save_all_weights(ner_model, filename)

    with open("index/word2index.pickle", "wb") as handle:
        pickle.dump(word2index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("index/index2tag.pickle", "wb") as handle:
        pickle.dump(index2tag, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("done.")

def main(args):
    
    parser = argparse.ArgumentParser(description="Named Entity Recognition (NER) model using Conditional Random Fields (CRFs). \
                                     This work is licensed \
                                     under the Creative Commons Attribution \
                                     4.0 International License.")
    
    parser.add_argument("--lr", help="Learning rate.", required=False, default=0.001, type=float)
    parser.add_argument("--verbose", help="Verbosity of the model fitting phase.", required=False, 
                        default=2, type=int, choices=[0,1,2])
    parser.add_argument("--max_epochs", help="Maximum number of training epochs.", required=False, default=5, type=int)
    parser.add_argument("--batch_size", help="Number of samples in a training batch.", required=False, default=256, type=int)
    parser.add_argument("--test_size", help="Size of the test set [0,1).", required=False, default=0.1, type=float)
    parser.add_argument("--lstm_units", help="Dimensionality of the LSTM output space.", required=False, default=50, type=int)
    parser.add_argument("--dropout", help="Fraction of the LSTM units to drop for the linear transformation of the recurrent state [0,1].",
                        required=False, default=0.1, type=float)
    parser.add_argument("--dense_units", help="Number of fully connected units for each temporal slice.", required=False, default=100, type=int)
    parser.add_argument("--dense_embedding", help="Dimension of the dense embedding.", required=False, default=50, type=int)
    parser.add_argument('--plot', dest="plot", help="Generate a plot of the loss & accuracy metrics.", action="store_true")
    parser.set_defaults(plot=False)
    
    args = parser.parse_args()
    
    # Load the dataset
    data_df = load_data()
    
    # Build the word & tag indices
    (word2index, index2word, tag2index, index2tag), sentences = pre_process(data_df)
    
    # Vectorize all data
    X, y = vectorize(word2index, index2word, tag2index, index2tag, sentences)
    
    # Split into training/test sets
    X_train, X_test, y_train, y_test = get_training_sets(X, y, test_size=args.test_size)
    
    # Build the model
    WORD_COUNT = len(index2word)
    
    ner_model = get_model(X_train, y_train, WORD_COUNT, lr=args.lr, LSTM_UNITS=args.lstm_units, LSTM_DROPOUT=args.dropout,
                         DENSE_UNITS=args.dense_units, DENSE_EMBEDDING=args.dense_embedding)
    
    # Train the model
    print("Model training....")
    
    history = ner_model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.max_epochs, verbose=args.verbose)
    
    if args.plot:
        plot_history(history.history)
        plt.savefig("model/history.png")
        
    print("Model training completed.")
    
    # Persist the model and indices
    save_model(ner_model, "model/ner_model.h5", word2index, index2tag)
    
    MAX_SENTENCE = X_train.shape[1]
    TAG_COUNT = y_train.shape[2]
    
    model_settings = {
      "DENSE_EMBEDDING" : args.dense_embedding,
      "LSTM_UNITS" : args.lstm_units,
      "LSTM_DROPOUT" : args.dropout,
      "DENSE_UNITS" : args.dense_units,
      "TAG_COUNT" : TAG_COUNT,
      "MAX_SENTENCE" : MAX_SENTENCE
    }
    
    with open("model/model_settings.pickle", "wb") as f:
        pickle.dump(model_settings, f, pickle.HIGHEST_PROTOCOL)
    
if __name__=='__main__':
    main(sys.argv)
