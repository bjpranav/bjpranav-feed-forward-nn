# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:31:40 2019

@author: Pranav Krishna
"""

#Loading Libraries
import os
from nltk.tokenize import word_tokenize
import re
import nltk
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
import warnings
import time
start_time = time.time()
warnings.filterwarnings("ignore")


def normalize_case(s):
    '''
    Paramaeter: Word to be normalized
    Converts words with capitalized first letters in to lower case.
    '''
    if (not s.isupper()):
        return s.lower()
    else:
        return s


def remove_tags(s):
    '''
    Paramaeter: Word to be normalized
    Removes HTML tags
    '''
    s = re.sub(r'[^\w\s]', '', s)
    return s


def count_words(rootdir):
    '''
    Parameter: root directory

    The funtion collects the training files. Tokenizes text into words. Creates stemmed vocabulary and
    Counts the the occurance of each word in each class(positve and negative).
    '''
    vocab = []

    bigram = []
    cleaned_document = []
    # For each directory in the path
    z = 0
    for subdir, dirs, files in os.walk(rootdir):
        # For each file in the directory
        for file in files:
            z = z + 1
            if (z > 1000):
                break
            cleaned_document = []
            f = open(rootdir + file, 'r')  # use the absolute URL of the file
            lines = f.readlines()
            # For each line in the file
            for line in lines:
                document = word_tokenize(line)
                for i in range(0, len(document)):
                    # Normalize case for the word
                    document[i] = normalize_case(document[i])
                    # Remove HTML tags
                    document[i] = remove_tags(document[i])
                    if (document[i] != ''):
                        cleaned_document.append(document[i])
                        vocab.append(document[i])

            # Store as bigrams
            bigram.extend(list(nltk.bigrams(cleaned_document)))

    # return positive bigram and vocabulary
    return bigram, vocab


def make_bigrams():
    '''
    The function collects positive bigrams, creates negative bigrams from them
    and stores them in a csv file
    '''
    rootdir = r'data\\'
    pos_bigram, vocab = count_words(rootdir)

    # Randomly creates 2 negative bigrams for each positive bigrams
    neg_bigram = []
    for bigram in pos_bigram:
        rand_neg = (bigram[0], random.choice(vocab))
        neg_bigram.append(rand_neg)
        rand_neg = (bigram[0], random.choice(vocab))
        neg_bigram.append(rand_neg)

    # COmbines the bigrams and stores it in a csv
    pos_df = pd.DataFrame(pos_bigram, columns=['first_word', 'second_word'])
    pos_df['tag'] = "pos"

    neg_df = pd.DataFrame(neg_bigram, columns=['first_word', 'second_word'])
    neg_df['tag'] = "neg"

    df = pos_df.append(neg_df, ignore_index=True)

    df.to_csv('bigrams.csv')

def buildModel():
    df = pd.read_csv('bigrams.csv')

    # Extracting Input and Output variables
    y = df['tag']
    X = df['first_word'] + ' ' + df['second_word']

    # COnverting text data into vectorized format(Count Vectorizer)
    X1 = CountVectorizer(stop_words='english')
    X1.fit(X)
    data = X1.transform(X)

    # Label Encoding the data
    le = preprocessing.LabelEncoder()
    le.fit(y)
    Y = le.transform(y)

    # Splitiing data into train and test data (20% test data)
    train, test, y_train, y_test = train_test_split(data, Y, test_size=0.2, random_state=1000)

    input_dim = train.shape[1]  # Number of features
    # Feed forward Neural Networks with 20 hidden layers and sigmoid Activation
    model = Sequential()
    model.add(layers.Dense(20, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    # Using TensorFlow backend.

    # Model with binary_crossentropy as loss function and learning rate of 0.001
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.00001)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()

    # Model with binary_crossentropy as loss function and learning rate of 0.001
    # from keras import optimizers
    # sgd = optimizers.SGD(lr=0.00001)
    # model.compile(loss='mse', optimizer= 'sgd',metrics=['accuracy'])
    # model.summary()

    history = model.fit(train, y_train,
                        epochs=2,
                        verbose=False,
                        validation_data=(test, y_test))

    loss, accuracy = model.evaluate(train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

if __name__ == "__main__" :
    make_bigrams()
    buildModel()
    print("--- %s seconds ---" % (time.time() - start_time))