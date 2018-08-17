# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:55:33 2017

@author: Rudrajit Aditya Akash
"""


import numpy as np
import nltk
from sklearn.neural_network import MLPClassifier
from feature_extraction import get_text_and_summary
from feature_extraction import get_input_and_ouptut_vectors
from feature_extraction import generate_summary_file
from gensim.models.keyedvectors import KeyedVectors
import pdb

def get_sentences(file_name):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    fp = open(file_name)
    text = fp.read()
    sentences = tokenizer.tokenize(text)
    return sentences

def get_total_word_count():
    f = open('git_prob.txt', 'r',encoding="ISO-8859-1")
    N = 0
    for line in f.readlines():
        line = str(line.lower())
        line = line.strip().lower()
        word_and_prob = str(line)
        count = word_and_prob[line.index(' ')+1:]
        N = N+int(count)
    f.close()
    # if the word was not found in the list, return 0
    return N

def train_and_test():
    #Get the actual and the summary text
    #pdb.set_trace()
    model = KeyedVectors.load_word2vec_format('/home/audy/GoogleNews-vectors-negative300.bin', binary=True)  
    [text,text_summ] = get_text_and_summary()
    #Generate the input and the output for the NN from the text and the summary.
    no = get_total_word_count()
    [X_train,Y_train] = get_input_and_ouptut_vectors(text,text_summ,2,1500,1,model,no)
    #X_train = np.load('X_train.npy')
    #Y_train = np.load('Y_train.npy')
    #Y_train = Y_train.astype(np.int)
    print(sum(Y_train))
    #print(Y_train[0:30])
    """
    Y_train_ones = Y_train[np.where(Y_train==1)]
    X_train_ones = np.squeeze(X_train[np.where(Y_train==1),:])
    Y_train_zeros = Y_train[np.where(Y_train==0)]
    X_train_zeros = np.squeeze(X_train[np.where(Y_train==0),:])
    prng = np.random.RandomState(954)
    #print(len(Y_train_ones))
    #print(X_train_zeros.shape)
    idx = prng.choice(X_train_zeros.shape[0],5*len(Y_train_ones),replace=False)
    X_train_zeros_random = np.squeeze(X_train_zeros[idx,:])
    X_train2 = np.vstack((X_train_ones,X_train_zeros_random))
    Y_train_zeros_random = np.squeeze(Y_train_zeros[idx])
    Y_train2 = np.concatenate((Y_train_ones,Y_train_zeros_random))
    idx2 = prng.permutation(len(Y_train2))
    X_train2 = X_train2[idx2,:]
    Y_train2 = Y_train2[idx2]
    """
    """
    m = 300
    win = 3
    w = (2*win+1)*m
    """
    #clf = MLPClassifier(activation='tanh', alpha=1e-5, hidden_layer_sizes=(w,w), random_state=1)
    #clf.fit(X_train2,Y_train2)
    [X_test,Y_test] = get_input_and_ouptut_vectors(text,text_summ,1501,2000,2,model,no)
    #X_test = np.load('X_test.npy')
    #Y_test = np.load('Y_test.npy')
    """
    Y_test = Y_test.astype(np.int)
    Y_test_predicted = clf.predict(X_test)
    print(sum(Y_test_predicted))
    test_accuracy = clf.score(X_test,Y_test)
    print(test_accuracy)
    generate_summary_file(text,Y_test_predicted,501,)
    Y_test_ones = Y_test[np.where(Y_test==1)]
    Y_test_predicted_ones = Y_test_predicted[np.where(Y_test==1)]
    diff = np.abs(Y_test_ones - Y_test_predicted_ones)
    print(sum(diff)/sum(Y_test_ones))
    print(sum(Y_test))
    """
    