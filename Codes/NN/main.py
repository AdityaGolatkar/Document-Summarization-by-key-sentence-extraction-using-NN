# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:55:33 2017

@author: Rudrajit Aditya Akash
"""

#from train2 import train_and_test
from train_ensemble_python import train_and_test
from train_ensemble_python import get_total_word_count
from feature_extraction import generate_summary_file
from feature_extraction import get_text_and_summary
from feature_extraction import get_text_and_summary2
from feature_extraction import get_input_for_CNN
from gensim.models.keyedvectors import KeyedVectors
from feature_extraction import get_input_and_ouptut_vectors
import numpy as np
#import nltk

train_and_test()

#[text,text_summ] = get_text_and_summary()
#key_sentence_labels = np.zeros(100)
#key_sentence_labels[1] = 1
#key_sentence_labels[8] = 1
#key_sentence_labels[17] = 1
#key_sentence_labels[26] = 1
#key_sentence_labels[35] = 1
#generate_summary_file(text,key_sentence_labels,3,15)

#X = np.ones([2,2100])
#X_CNN = get_input_for_CNN(X)
#print(X_CNN[:,0,:].shape)

#no = get_total_word_count()
# no = 3484135472
#print(no)
#model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  
#[text,text_summ] = get_text_and_summary2()
#[X_train,Y_train] = get_input_and_ouptut_vectors(text,text_summ,0,3,1,model,no)