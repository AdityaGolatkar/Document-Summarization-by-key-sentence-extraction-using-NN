# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:55:33 2017

@author: Rudrajit Aditya Akash
"""

import csv
import nltk
#nltk.download('punkt')
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pdb
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import TruncatedSVD



def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX

def get_sentences(file_name):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    fp = open(file_name)
    text = fp.read()
    sentences = tokenizer.tokenize(text)
    return sentences

def get_words(sentence):
    words = re.sub("[^\w]", " ",  sentence).split()
    return words

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

def get_one_word_prob(word_ex,N):
    f = open('git_prob.txt', 'r',encoding="ISO-8859-1")
    for line in f.readlines():
        line = str(line.lower())
        line = line.strip().lower()
        word_and_prob = str(line)
        if(len(word_and_prob) > len(word_ex)):
            if(word_and_prob[len(word_ex)] == ' '):
                word = word_and_prob[0:len(word_ex)]
                prob = word_and_prob[len(word_ex)+1:]
                if word.lower() == word_ex.lower():
                    prob = (int(prob)/N)
                    f.close()
                    return prob
    # if the word was not found in the list, return 0
    return 0

def get_word_prob(words,N):
    word_prob = np.zeros(len(words))
    for i in range(0,len(words)):
        word_prob[i] = get_one_word_prob(words[i],N)                        
    return word_prob

def prob_and_vec(words,N):   
    m = 300
    tot = 100000000000
    word_vec = np.zeros([m,len(words)])
    word_prob = np.zeros(len(words))
    word_prob = get_word_prob(words,N)
    #pdb.set_trace()
    for i in range(0,len(words)):
        try:
            word_vec[:,i] = model[words[i]]
            #word_prob[i] = model.vocab[words[i]].count/tot
        except:
            try:
                word_vec[:,i] = model[words[i].capitalize()]
               	#word_prob[i] = model.vocab[words[i].capitalize()].count/tot
            except:
                try:
                    word_vec[:,i] = model[words[i].upper()]
                    #word_prob[i] = model.vocab[words[i].upper()].count/tot  
                except:     
                    word_vec[:,i] = np.zeros((300,))
                    #word_prob[i] = 0                       
    return word_vec, word_prob

def get_sentence_embeddings(file_name,N):
    sentences = get_sentences(file_name)
    m = 300
    sentence_embedding = np.zeros([m,len(sentences)])
    #a = 50
    a = 8e-3
    pdb.set_trace()
    #sentence_embedding[] = sentence_embedding - np.mean(sentence_embedding[:,i])
    for i in range(0,len(sentences)):
        words = get_words(sentences[i])
        [words_vec,words_prob] = prob_and_vec(words,N)
        #words_prob = 1e5*words_prob
        words_coeff =  np.divide(a,a+words_prob)
        if (len(words) != 0):
            sentence_embedding[:,i] = np.dot(words_vec,words_coeff)/len(words)
    #sentence_embedding_trans = sentence_embedding.T
    #sentence_embedding_mean = sentence_embedding_trans.mean(axis = 0)
    #sentence_embedding_centered = sentence_embedding_trans - sentence_embedding_mean
    #sentence_embedding_centered = sentence_embedding_centered.T
    #sentence_embedding_mean = sentence_embedding_mean.T

    #sentence_embedding_mean = sentence_embedding.mean(axis = 1)
    #sentence_embedding_mean = np.reshape(sentence_embedding_mean,[m,1])
    #sentence_embedding_mean = np.tile(sentence_embedding_mean,len(sentences))
    #sentence_embedding_centered = sentence_embedding - sentence_embedding_mean

    #[U,_,_] = np.linalg.svd(sentence_embedding_centered)
    #u = np.reshape(U[:,0],(len(U[:,0]),1))

    #sentence_embedding_centered = sentence_embedding_centered - np.dot(u,np.dot(u.transpose(),sentence_embedding_centered))
    #sentence_embedding = sentence_embedding_centered + sentence_embedding_mean

    sentence_embedding_pure = remove_pc(sentence_embedding.T,1)
    sentence_embedding_pure = sentence_embedding_pure.T

    return sentence_embedding_pure

def get_key_sentence_labels(sentence_embeddings,sentence_embeddings_summ):
    dim_summ = sentence_embeddings_summ.shape
    summ_ct = dim_summ[1]
    dim = sentence_embeddings.shape
    ct = dim[1]
    key_sentences = np.zeros(ct)
    for i in range(0,summ_ct):
        dot_prod = cosine_similarity((sentence_embeddings_summ[:,i]).reshape(1,-1),np.transpose(sentence_embeddings))
        key_sentences[np.argmax((dot_prod))] = 1
        print(np.argmax((dot_prod)))
        pdb.set_trace()
    return key_sentences

fname = 'quora.txt'
sent = get_sentences(fname)
pdb.set_trace()
model = KeyedVectors.load_word2vec_format('/home/audy/GoogleNews-vectors-negative300.bin', binary=True)  
N = get_total_word_count()
embed = get_sentence_embeddings(fname,N)
key_sen = get_key_sentence_labels(embed,embed)
