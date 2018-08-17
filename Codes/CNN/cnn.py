# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:55:33 2017

@author: Rudrajit Aditya Akash
"""


import pdb
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, LocallyConnected2D, BatchNormalization
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from feature_extraction import generate_summary_file
from feature_extraction import get_text_and_summary


import numpy as np

np.random.seed(2)
# dimensions of our images.
def get_input_for_CNN(X):
    num_points = X.shape[0]
    win = 3
    w = (2*win+1)
    m = 300
    X_CNN = np.zeros([num_points,m,w,1])
    for i in range(0,w):
        X_CNN[:,:,i,0] = X[:,m*i:m*(i+1)]
    return X_CNN

[text,text_summ] = get_text_and_summary()

X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')
X_test = np.load('X_test.npy')
X_test = get_input_for_CNN(X_test)
Y_test = np.load('Y_test.npy')
Y_train_ones = Y_train[np.where(Y_train==1)]
X_train_ones = np.squeeze(X_train[np.where(Y_train==1),:])
Y_train_zeros = Y_train[np.where(Y_train==0)]
X_train_zeros = np.squeeze(X_train[np.where(Y_train==0),:])
prng = np.random.RandomState(954)
#print(len(Y_train_ones))
#print(X_train_zeros.shape)
idx = prng.choice(X_train_zeros.shape[0],int(3*len(Y_train_ones)),replace=False)
X_train_zeros_random = np.squeeze(X_train_zeros[idx,:])
X_train2 = np.vstack((X_train_ones,X_train_zeros_random))
Y_train_zeros_random = np.squeeze(Y_train_zeros[idx])
Y_train2 = np.concatenate((Y_train_ones,Y_train_zeros_random))
idx2 = prng.permutation(len(Y_train2))
X_train2 = X_train2[idx2,:]
Y_train2 = Y_train2[idx2]
X_train2_cnn = get_input_for_CNN(X_train2)


img_width, img_height = 7,300 
epochs = 20
batch_size_train = 32
batch_size_test = 9

fname = "wts2.h5",

print(X_train2_cnn.shape)
#input_shape = (1, img_width, img_height)
input_shape = (img_height, img_width,1)

"""
model = Sequential()
layer1 = Conv2D(200, (300, 2),input_shape=input_shape)
model.add(layer1)
print(layer1.input_shape)
print(layer1.output_shape)
model.add(Activation('tanh'))
layer2 = MaxPooling2D(pool_size=(1,6),data_format="channels_last")
model.add(layer2)
print(layer2.input_shape)
print(layer2.output_shape)
model.add(Flatten())

"""
"""

model.add(Dense(200))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
"""
model = Sequential()

#model.add(Conv2D(32, (5, 5),padding='same',input_shape=input_shape))
#model.add(Conv2D(32, (5, 5),padding='same',input_shape=input_shape))

model.add(Conv2D(128, (3, 3),padding='same',input_shape=input_shape))

#model.add(Conv2D(128, (3, 3),padding='same',input_shape=input_shape))
model.add(Activation('tanh'))
layer1 = MaxPooling2D(pool_size=(2,2),strides=(2,2),data_format="channels_last")
model.add(layer1)
print(layer1.input_shape)
print(layer1.output_shape)
model.add(Flatten())

"""
model.add(Dense(150))
model.add(Activation('tanh'))
"""

model.add(Dense(256))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#model.load_weights(fname)

checkpointer = ModelCheckpoint(filepath= '/home/audy/Desktop/Drive/IIT/ML/Project/wts1.h5', monitor='val_acc',verbose=0, save_best_only=True,save_weights_only=True)


model.fit(
    X_train2_cnn,
    Y_train2,
    epochs = epochs,
    verbose =1,
    callbacks=[checkpointer],
    validation_split = 0.2,
    shuffle = False
    )

model.save_weights('/home/audy/Desktop/Drive/IIT/ML/Project/wts2.h5')


predictions = model.predict(X_test,batch_size = batch_size_test,verbose = 1)
predictions = np.around(predictions).astype(int)
result = model.evaluate(X_test,Y_test,batch_size = batch_size_test,verbose = 1)
print('\n')
print('Accuracy is :')
print(result[1])

Y_test_predicted = predictions
print("thirlast",sum(Y_test_predicted))
generate_summary_file(text,Y_test_predicted,1501,2000)
Y_test_ones = Y_test[np.where(Y_test==1)]
Y_test_predicted_ones = Y_test_predicted[np.where(Y_test==1)]
pdb.set_trace()
Y_test = np.reshape(Y_test,Y_test_predicted.shape)
Y_test_ones = np.reshape(Y_test_ones,Y_test_predicted_ones.shape)
diff = np.abs(Y_test_ones - Y_test_predicted_ones)
print("seclast",sum(diff)/sum(Y_test_ones))
print("last",sum(Y_test))

