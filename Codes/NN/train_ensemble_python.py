# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:55:33 2017

@author: Rudrajit Aditya Akash
"""

import numpy as np
import nltk
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import preprocessing
from feature_extraction import get_text_and_summary
from feature_extraction import get_text_and_summary2
from feature_extraction import get_input_and_ouptut_vectors
from feature_extraction import generate_summary_file
from gensim.models.keyedvectors import KeyedVectors


def get_total_word_count():
    #f = open('git_prob.txt', 'r',encoding="ISO-8859-1")
    f = open('git_prob.txt', 'r')
    N = 0
    for line in f.readlines():
        line = str(line.lower())
        line = line.strip().lower()
        word_and_prob = str(line)
        count = word_and_prob[line.index(' ')+1:]
        N = N+int(count)
    f.close()
    return N

def train_and_test():
    #Get the actual and the summary text
    #pdb.set_trace()
    
    #print(1)
    #model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  
    [text,text_summ] = get_text_and_summary()
    #Generate the input and the output for the NN from the text and the summary.

    #no = get_total_word_count()

    #[X_train,Y_train] = get_input_and_ouptut_vectors(text,text_summ,0,250,1,model,no)
    X_train = np.load('/home/audy/Desktop/Drive/IIT/ML/Project/X_train.npy')
    """
    Xt_train = np.zeros([X1_train.shape[0]-1,X1_train.shape[1]])
    Xt_train[0:40,:] = X1_train[0:40,:]
    Xt_train[40:,:] = X1_train[41:,:]
    X1_train = Xt_train
    """
    Y_train = np.load('/home/audy/Desktop/Drive/IIT/ML/Project/Y_train.npy')
    """
    Yt_train = np.zeros(len(Y1_train)-1)
    Yt_train[0:40] = Y1_train[0:40]
    Yt_train[40:] = Y1_train[41:]
    Y1_train = Yt_train
    """
    """
    X2_train = np.load('X_train22.npy')
    Xt_train = np.zeros([X2_train.shape[0]-1,X2_train.shape[1]])
    Xt_train[0:183,:] = X2_train[0:183,:]
    Xt_train[183:,:] = X2_train[184:,:]
    X2_train = Xt_train
    Y2_train = np.load('Y_train22.npy')
    Yt_train = np.zeros(len(Y2_train)-1)
    Yt_train[0:183] = Y2_train[0:183]
    Yt_train[183:] = Y2_train[184:]
    Y2_train = Yt_train
    
    X_train = np.vstack((X1_train,X2_train))
    Y_train = np.append(Y1_train,Y2_train,axis=0)
    """
    Y_train = Y_train.astype(np.int)
    print(sum(Y_train))
    #print(Y_train[0:30])
    
    #[X_test,Y_test] = get_input_and_ouptut_vectors(text,text_summ,251,295,2,model,no)
    X_test = np.load('/home/audy/Desktop/Drive/IIT/ML/Project/X_test.npy')
    Y_test = np.load('/home/audy/Desktop/Drive/IIT/ML/Project/Y_test.npy')
    Y_test = Y_test.astype(np.int)
    print(sum(Y_test))
    
    Y_train_ones = Y_train[np.where(Y_train==1)]
    X_train_ones = np.squeeze(X_train[np.where(Y_train==1),:])
    Y_train_zeros = Y_train[np.where(Y_train==0)]
    X_train_zeros = np.squeeze(X_train[np.where(Y_train==0),:])
    m = 300
    win = 3
    w = (2*win+1)*m
    p = np.ceil(X_train_ones.shape[0]/10)
    p = p.astype(np.int)
    Y_test_predicted = np.zeros(X_test.shape[0])
    Q = 15
    N = 0
    for i in range(0,Q):
        prng = np.random.RandomState((i+1)**2)
        idx1 = prng.choice(X_train_ones.shape[0],p,replace=False)
        X_train_ones_random = np.squeeze(X_train_ones[idx1,:])
        idx0 = prng.choice(X_train_zeros.shape[0],int(2*p),replace=False)
        X_train_zeros_random = np.squeeze(X_train_zeros[idx0,:])
        X_train2 = np.vstack((X_train_ones_random,X_train_zeros_random))
        Y_train_ones_random = np.squeeze(Y_train_ones[idx1])
        Y_train_zeros_random = np.squeeze(Y_train_zeros[idx0])
        Y_train2 = np.concatenate((Y_train_ones_random,Y_train_zeros_random))
        idx2 = prng.permutation(len(Y_train2))
        X_train2 = X_train2[idx2,:]
        Y_train2 = Y_train2[idx2]
        #Xn_train2 = preprocessing.scale(X_train2)
        clf = MLPClassifier(activation='tanh', alpha=1e-5, hidden_layer_sizes=(w,w), random_state=1)
        clf.fit(X_train2,Y_train2)
        sc = clf.score(X_test,Y_test)
        print(sc)
        if (sc > 0.5):
            Y_test_predicted = Y_test_predicted + clf.predict(X_test)
            N = N + 1
        #Y_test_predicted = Y_test_predicted + sc*clf.predict(X_test)
        #N = N + sc
        print(i)
    #print(N)
    Y_test_predicted = Y_test_predicted/N
    Y_test_predicted = np.around(Y_test_predicted)
    Y_test_predicted = Y_test_predicted.astype(np.int)
    print(sum(Y_test_predicted))
    #test_accuracy = clf.score(X_test,Y_test)
    #print(test_accuracy)
    generate_summary_file(text,Y_test_predicted,1501,2000)
    Y_test_ones = Y_test[np.where(Y_test==1)]
    Y_test_predicted_ones = Y_test_predicted[np.where(Y_test==1)]
    diff = np.abs(Y_test_ones - Y_test_predicted_ones)
    print(sum(diff)/sum(Y_test_ones))
    
    