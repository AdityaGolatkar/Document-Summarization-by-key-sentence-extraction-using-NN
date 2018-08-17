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



def ensemble_nn(x_train,y_train,x_test,y_test):
	model = Sequential()
	model.add(Dense(units=2100,input_shape=(2100,)))
	model.add(Activation('tanh'))
	#model.add(Dropout(0.5))
	#model.add(Dense(units=2100))
	#model.add(Activation('tanh'))
	#model.add(Dropout(0.5))
	model.add(Dense(units=1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy',
					optimizer='rmsprop',
					metrics=['accuracy'])
	model.fit(x_train,y_train,epochs=10,batch_size=10,validation_split=0.2,verbose=0)
	loss_and_metrics = model.evaluate(x_test,y_test,verbose=0)
	print("Accuracy : ")
	acc = loss_and_metrics[1]
	print(acc)
	prediction = model.predict(x_test,verbose=0)
	return acc,prediction




