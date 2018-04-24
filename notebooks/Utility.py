
# coding: utf-8

# In[1]:


from math import sqrt

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold

import keras
from keras import models, layers, callbacks, constraints, backend, activations
from keras.models import model_from_json


from matplotlib import pyplot
import matplotlib.patches as mpatches

import datetime
import pathlib
import time
import json
import sys
import os

get_ipython().magic('matplotlib inline')
pd.options.display.max_columns = None


# In[2]:


def string_time(start, end):
    total = end - start
    hour = 60 * 60
    minute = 60
    
    num = total
    rem = num % hour
    
    hours = (num - rem) / hour

    num = rem
    rem = num % minute

    minutes = (num - rem) / minute
    
    return '{:.0f}:{:.0f}:{:.0f}'.format(hours, minutes, rem)


# In[3]:


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)

    # transform train
    train_scaled = scaler.transform(train)
    # transform test
    test_scaled = scaler.transform(test)

    return scaler, train_scaled, test_scaled


# In[4]:


# inverse scaling for a forecasted value
def invert_scale(scaler, X, y):
    
    # combine data back into single matrix for scaling
    features_and_labels_scaled = np.append(X, y, axis=1)
    features_and_labels = scaler.inverse_transform(features_and_labels_scaled)
    
    return features_and_labels


# In[5]:


# create and ann_model to be used with sklearn wrapper
def ann_model(
    features = 1,
    output_size = 1,
    batch_size = 1,
    hidden_layers = [{'neurons': 50, 'dropout': 0.1 }],
    activation = ['relu', 'softmax'],
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    noise = (False, 0.07),
    metrics = ['accuracy']
):
              
    # create sequenital model
    
    inputs = layers.Input(shape=(features,))
    
    count = 1
    x = None
    previous = inputs
    
    # makes every hidden layer
    for i in range(len(hidden_layers)):

        layer = hidden_layers[i]
        
        if noise[0]:
            
            x = layers.GaussianNoise(noise[1])(previous)
            previous = x

        x = layers.Dense(
            units = layer['neurons'],
            activation = activation[0],
            name='ANN_hidden_layer_{}'.format(i + 1)
        )(previous)
        
        x = layers.Dropout(
            rate = layer['dropout']
        )(x)
        previous = x
           
    # makes output layer
    outputs = layers.Dense(
        units = output_size,
        activation = activation[1],
        name = 'output_layer',
    )(previous)
    
    model = models.Model(
        inputs = inputs, 
        outputs = outputs
    )
      
    model.compile(
        loss = loss,
        optimizer = optimizer,
#         metrics = metrics,
        metrics=["accuracy"],
    )

    return model


# In[6]:


def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels = [model(model_input) for model in models] 

    add_layer = layers.Average()(yModels)


    # build model from same input and avg output
    modelEns = layers.Model(inputs=model_input, outputs= add_layer,    name='ensemble')  
   
    return modelEns

