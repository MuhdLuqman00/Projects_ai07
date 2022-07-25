# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:18:09 2022

@author: muham
"""

# 1. Import packages 

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import os, datetime
from sklearn.impute import SimpleImputer 

#%%

# 2. Data preparation 

file_path = r"C:\Users\muham\Documents\dekstop\AI Machine Learning\Deep Learning with Python\Project 1 - Heart Disease\heart.csv"
HD_dataset = pd.read_csv(file_path, header = 0)

#%%
# check for missing values (no missing values can be seen)

print(HD_dataset.isna().sum())
#%%
# Separate labels from features 
data_features = HD_dataset.copy()
data_labels = data_features.pop('target')

# make sure that the data is in the correct format (0,1)

#%%
# check the data types 

print(data_features.dtypes)
# data_features['sex'] = pd.Categorical(data_features.sex)
# data_features['cp'] = pd.Categorical(data_features.cp)
# data_features['restecg'] = pd.Categorical(data_features.restecg)
# data_features['ca'] = pd.Categorical(data_features.ca)

#%%

# data splitting 

SEED = 12345

(x_train,  x_test, y_train, y_test) = train_test_split(data_features, data_labels, test_size = 0.2, random_state = SEED)

#%%
# Normalize the data

standardizer = StandardScaler()
standardizer.fit(x_train)
x_train = standardizer.transform(x_train)
x_test = standardizer.transform(x_test)

# After the data is normalized, it is returned in a the format of ndarray
#%%
# Building the model 

nclass = len(np.unique(y_test))
nIn = x_test.shape[1]

l2 = keras.regularizers.L2(l2 = 0.01)


model = keras.Sequential()
model.add(layers.InputLayer(input_shape = (nIn,)))
model.add(layers.Dense(128, activation = 'relu', kernel_regularizer = l2))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation = 'relu',  kernel_regularizer = l2))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(32, activation = 'relu',  kernel_regularizer = l2))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(nclass, activation = 'softmax'))

#%%

model.summary()

#%%
# compile the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#%%
#Early stopping callback
es = EarlyStopping(patience=10,verbose=1,restore_best_weights=True)
#TensorBard callback
base_log_path = r"C:\Users\muham\Documents\dekstop\AI Machine Learning\Deep Learning with Python\tb_logs"
log_path = os.path.join(base_log_path, 'Project 1', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = TensorBoard(log_dir=log_path)

#%%
#training the model

BATCH_SIZE = 32
EPOCHS = 100
history = model.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[tb,es])