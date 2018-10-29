#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 18:49:48 2018

@author: adriano
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import urllib
import scipy.io.wavfile
import pydub
import os, glob
from numpy import fft as fft
from sklearn.tree import DecisionTreeClassifier
import librosa
import librosa.display
import keras
from keras.layers import Activation, Dense, Dropout, Conv2D,Flatten, MaxPooling2D
from keras.models import Sequential
import random
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import warnings


#CRIAÇÃO DA DATASET
dataset = []

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/cel/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
    ps=librosa.feature.melspectrogram(y=y,sr=sr)
    dataset.append((ps,0))#cel
    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/cla/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
    ps=librosa.feature.melspectrogram(y=y,sr=sr)
    dataset.append((ps,1))#cla

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/flu/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
    ps=librosa.feature.melspectrogram(y=y,sr=sr)
    dataset.append((ps,2))#flu

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/gac/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
    ps=librosa.feature.melspectrogram(y=y,sr=sr)
    dataset.append((ps,3))#gac

temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/gel/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
    ps=librosa.feature.melspectrogram(y=y,sr=sr)
    dataset.append((ps,4))#gel
    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/org/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
    ps=librosa.feature.melspectrogram(y=y,sr=sr)
    dataset.append((ps,5))#org
    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/pia/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
    ps=librosa.feature.melspectrogram(y=y,sr=sr)
    dataset.append((ps,6))#pia
    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/sax/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
    ps=librosa.feature.melspectrogram(y=y,sr=sr)
    dataset.append((ps,7))#sax
    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/tru/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
    ps=librosa.feature.melspectrogram(y=y,sr=sr)
    dataset.append((ps,8))#tru
    
temp_folder="/home/adriano/Área de Trabalho/Treino/IRMAS-TrainingData_red/vio/"

os.chdir(temp_folder)

listaDeAudios = []

for file in glob.glob("*.wav"):
    listaDeAudios.append(file)
    
for i in range(len(listaDeAudios)):
    y, sr = librosa.load((temp_folder+listaDeAudios[i]),duration = 3)
    ps=librosa.feature.melspectrogram(y=y,sr=sr)
    dataset.append((ps,9))#vio
    
        
    
###############################################    
###############################################
    
def creatCallbacks(nameModel):

    weight_path ="{}_weights.best.hdf5".format(nameModel)

    checkpoint = ModelCheckpoint(weight_path, monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max', save_weights_only = True)

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='min', 
                                       epsilon=0.0001, cooldown=3, min_lr=0.0001)

    callbacks_list = [checkpoint, reduceLROnPlat]

    return weight_path, callbacks_list





    
D = dataset
random.shuffle(D)


train = D[:1237]
test = D[1237:]


X_train,y_train = zip(*train)
X_test,y_test = zip(*test)

# Reshape for CNN input
X_train = np.array([x.reshape( (128, 130, 1) ) for x in X_train])
X_test = np.array([x.reshape( (128, 130, 1) ) for x in X_test])


# One-Hot encoding for classes
y_train = np.array(keras.utils.to_categorical(y_train, 10))
y_test = np.array(keras.utils.to_categorical(y_test, 10))

model = Sequential()

input_shape=(128, 130, 1)

model.add(Conv2D(64, (5, 5), strides=(1, 1), kernel_initializer = 'glorot_normal',
                                             input_shape=input_shape))

model.add(Dropout(rate=0.5))

model.add(MaxPooling2D())
model.add(Activation('tanh'))

model.add(Conv2D(128, (5, 5),kernel_initializer = 'glorot_normal', padding="valid"))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('tanh'))
#model.add(Dropout(rate=0.5))

model.add(Conv2D(128, (5, 5), kernel_initializer = 'glorot_normal',padding="valid"))
model.add(Activation('tanh'))
#model.add(Dropout(rate=0.5))

model.add(Flatten())
model.add(Dropout(rate=0.5))

model.add(Dense(64,kernel_initializer = 'glorot_normal'))

model.add(Activation('tanh'))
model.add(Dropout(rate=0.5))

model.add(Dense(10,kernel_initializer = 'glorot_normal'))
model.add(Activation('softmax'))

model.compile(
	optimizer="rmsprop",
	loss="categorical_crossentropy",
	metrics=['accuracy'])


weight_path_model, callbacks_list_model = creatCallbacks('NOME_MODELO11')

model.load_weights(weight_path_model)

history = model.fit(x=X_train,y=y_train,epochs=20,
                              #batch_size=32,
                              validation_data=(X_test, y_test),
                              callbacks = callbacks_list_model)

#model.load_weights(weight_path_model)
model.save('NOME_MODELO10.h5')

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
Y_test = np.argmax(y_test, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
cm = confusion_matrix(Y_test, y_pred)
print(cm)  


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


    
    
    
    
    
    
    
    