# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:55:33 2017

@author: Rudrajit Aditya Akash
"""


import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
import numpy as np
import pdb

def get_input_for_CNN(X):
    num_points = X.shape[0]
    win = 3
    w = (2*win+1)
    m = 300
    X_CNN = np.zeros([num_points,m,w,1])
    for i in range(0,w):
        X_CNN[:,:,i,0] = X[:,m*i:m*(i+1)]
    return X_CNN

X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')
Y_train_ones = Y_train[np.where(Y_train==1)]
X_train_ones = np.squeeze(X_train[np.where(Y_train==1),:])
Y_train_zeros = Y_train[np.where(Y_train==0)]
X_train_zeros = np.squeeze(X_train[np.where(Y_train==0),:])
prng = np.random.RandomState(954)
#print(len(Y_train_ones))
#print(X_train_zeros.shape)
idx = prng.choice(X_train_zeros.shape[0],2*len(Y_train_ones),replace=False)
X_train_zeros_random = np.squeeze(X_train_zeros[idx,:])
X_train2 = np.vstack((X_train_ones,X_train_zeros_random))
Y_train_zeros_random = np.squeeze(Y_train_zeros[idx])
Y_train2 = np.concatenate((Y_train_ones,Y_train_zeros_random))
idx2 = prng.permutation(len(Y_train2))
X_train2 = X_train2[idx2,:]
Y_train2 = Y_train2[idx2]


x_train = X_train2
x_test = X_test
y_train = Y_train2
y_test = Y_test


model = Sequential()
model.add(Dense(units=3000,input_shape=(2100,)))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(units=100))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(units=1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
				optimizer='rmsprop',
				metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,batch_size=30,validation_split=0.3)

predictions = model.predict(x_test)
predictions = np.around(predictions).astype(int)
result = model.evaluate(X_test,Y_test,verbose = 1)
print('\n')
print('Accuracy is :')
print(result[1])

Y_test_predicted = predictions
print("thirlast",sum(Y_test_predicted))
#generate_summary_file(text,Y_test_predicted,1501,2000)
Y_test_ones = Y_test[np.where(Y_test==1)]
Y_test_predicted_ones = Y_test_predicted[np.where(Y_test==1)]
pdb.set_trace()
Y_test = np.reshape(Y_test,Y_test_predicted.shape)
Y_test_ones = np.reshape(Y_test_ones,Y_test_predicted_ones.shape)
diff = np.abs(Y_test_ones - Y_test_predicted_ones)
print("seclast",sum(diff)/sum(Y_test_ones))
print("last",sum(Y_test))





