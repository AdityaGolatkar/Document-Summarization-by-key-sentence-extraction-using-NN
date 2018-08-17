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
#model = KeyedVectors.load_word2vec_format('/home/audy/GoogleNews-vectors-negative300.bin', binary=True)  
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

def prob_and_vec(words,model,N):   
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

def get_sentence_embeddings(file_name,model,N):
    sentences = get_sentences(file_name)
    m = 300
    sentence_embedding = np.zeros([m,len(sentences)])
    #a = 50
    a = 8e-3
    #pdb.set_trace()
    #sentence_embedding[] = sentence_embedding - np.mean(sentence_embedding[:,i])
    for i in range(0,len(sentences)):
        words = get_words(sentences[i])
        [words_vec,words_prob] = prob_and_vec(words,model,N)
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



def get_text_and_summary():
    news = csv.reader(open('news_summary.csv',encoding="ISO-8859-1"), delimiter=',')
    N = 4515
    text = [[] for i in range(N)]
    text_summ = [[] for i in range(N)]
    i = 0
    b = 0
    for row in news:
        if b == 0:
            b = 1
            continue
        else:
            # To correct most of the bullshit in the actual text
            t = str(row[5])
            lt = len(t)
            for j in range(0,lt-1):
                if (j >= len(t)-1):
                    break
                if (t[j] == '.' and t[j+1] != ' '):
                    t1 = t[0:j+1]
                    t2 = t[j+1:]
                    t = t1+' '+t2
                if (t[j] == '?'):
                    t1 = t[0:j]
                    t2 = t[j+1:]
                    t = t1+t2
            text[i] = t
            text_summ[i] = str(row[4])
            #text[i] = row[5]
            #text_summ[i] = row[4]
            i = i+1
    return text,text_summ


def get_key_sentence_labels(sentence_embeddings,sentence_embeddings_summ):
    #Get the no of sentences in summary
    dim_summ = sentence_embeddings_summ.shape
    summ_ct = dim_summ[1]
    #Get the no of sentences in actual file
    dim = sentence_embeddings.shape
    ct = dim[1]
    key_sentences = np.zeros(ct)
    for i in range(0,summ_ct):
        #tmp = np.transpose(np.tile(sentence_embeddings_summ[:,i],(ct,1)))
        # inner product used to measure closeness
        #dot_prod = np.sum(np.multiply(tmp,sentence_embeddings),0)
        dot_prod = cosine_similarity((sentence_embeddings_summ[:,i]).reshape(1,-1),np.transpose(sentence_embeddings))
        key_sentences[np.argmax((dot_prod))] = 1
        #print(np.argmax((dot_prod)))
    return key_sentences
 
       
# Zero padding done to deal with sentences at the beginning and end
def get_input_and_ouptut_vectors(text,text_summ,N1,N2,z,model,no):
    N = 4515
    # m = 300 from fastText word vectors
    #m = 300
    # m = 100 for gensim
    m = 300
    win = 3
    w = (2*win+1)*m
    # X will be a p X w matrix, where p = number of sentences available
    # Each row contains the sentence embedding vectors (stacked one after the 
    # other) of the (2*win+1) sentences in a window - For Feedforward NN
    X = np.zeros([1,w])
    Y = np.zeros([1,2])
    b = 0
    for i in range(N1,N2):
        # clear the contents of the file before every iteration
        open('text.txt', 'w').close()
        file_text = open('text.txt','w') 
        file_text.write(text[i]) 
        file_text.close()
        sentences = get_sentences('text.txt')
        if (len(sentences) == 0):
            continue
        sentence_embeddings = get_sentence_embeddings('text.txt',model,no)
        #pdb.set_trace()
        open('text_summ.txt', 'w').close()
        file_summ = open('text_summ.txt','w') 
        file_summ.write(text_summ[i]) 
        file_summ.close()
        sentence_embeddings_sum = get_sentence_embeddings('text_summ.txt',model,no)
        # key_sentences should be a nX1 array (n = number of sentences) 
        # containing only 1s (if the sentence corresponding to that index is a 
        # key sentence) and 0s.
        key_sentences = get_key_sentence_labels(sentence_embeddings,sentence_embeddings_sum)
        
        #for t in range(0,len(key_sentences)):
        #    if(key_sentences[t] == 1):
        #        print(sentences[t])
        #        print('YaaY')
        
        #pdb.set_trace()
        # zero-pad 'sentence_embeddings' - not sure if this is a good idea!!!
        sentence_embeddings_zero_padded = np.zeros([m,2*win+sentence_embeddings.shape[1]])
        sentence_embeddings_zero_padded[:,win:win+sentence_embeddings.shape[1]] = sentence_embeddings
        sentence_embeddings = sentence_embeddings_zero_padded
        for j in range(win,len(key_sentences)+win):
            #if (j >= win and j < len(key_sentences)-win):
            if b == 0:
                X[b,:] = (np.transpose(sentence_embeddings[:,j-win:j+win+1])).ravel()
                Y[b,0] = key_sentences[j-win]
                b = 1
            else:
                X = np.vstack((X,(np.transpose(sentence_embeddings[:,j-win:j+win+1])).ravel()))
                Y = np.vstack((Y,np.array([key_sentences[j-win],0])))
        print(i)
    Y = Y[:,0]
    
    if z == 1:
       np.save('X_train.npy',X) 
       np.save('Y_train.npy',Y) 
    else:
       np.save('X_test.npy',X) 
       np.save('Y_test.npy',Y)        
    
    return X,Y
        

def generate_summary_file(text,key_sentence_labels,N1,N2):
    file_write_summ = open('summary.txt','w') 
    ct = 0
    for i in range(N1,N2):
        open('text.txt', 'w').close()
        file_text = open('text.txt','w') 
        file_text.write(text[i]) 
        file_text.close()
        sentences = get_sentences('text.txt')
        for j in range(0,len(sentences)):
            if ct<len(key_sentence_labels):   
                if (key_sentence_labels[ct] == 1):
                    file_write_summ.write(sentences[j])
                ct = ct+1
        file_write_summ.write('\n')
        file_write_summ.write('\n')
    file_write_summ.close()
        

    
    