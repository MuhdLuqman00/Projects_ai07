# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 09:20:02 2022

@author: muham
"""

# 0. Import Packages 

#1. Import packages
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
from sklearn import metrics
import matplotlib.pyplot as plt

#%%

# 1. Loading the data 

file_path = r"C:\Users\muham\Documents\dekstop\AI Machine Learning\Deep Learning with Python\Project 2 - Worker Productivity\garments_worker_productivity.csv"

dataset = pd.read_csv(file_path, header = 0)

#%%
# 2. Data Preparation 
'''
Check existence of missing values.

wip - 506 missing values
'''

print(dataset.isna().sum())

#%%
# 2.1 Dropping unwanted feature
dataset = dataset.drop(['date'], axis = 1)

#%%
# 2.2 One hot encode the categorical features 
dataset = pd.get_dummies(data = dataset)
#dataset.pop('wip')
#%%
# 2.3 Extracting the labels
data_labels = dataset.pop('actual_productivity').values

#%%
# 2.3 impute the missing values
imputer = SimpleImputer(strategy='mean')
dataset_imputed = imputer.fit_transform(dataset)

#%%
# 3. splitting the dataset
SEED = 12345
(x_interim, x_val, y_interim, y_val) = train_test_split(dataset_imputed, data_labels, test_size = 0.2, random_state = SEED)

(x_train, x_test, y_train, y_test) = train_test_split(x_interim, y_interim, test_size = 0.2, random_state = SEED)

#%%
# 4. Normalizing data

standardizer = StandardScaler()
standardizer.fit(x_train)
x_train = standardizer.transform(x_train)
x_val = standardizer.transform(x_val)
x_test = standardizer.transform(x_test)

#%%

# 5. Creating Model

nIn = x_test.shape[1]
inputs = keras.Input(shape = (nIn,))

#h1 = layers.Dense(512, activation = 'selu',kernel_initializer='normal') # to counter the dying relu.
#h2 = layers.Dense(512, activation = 'selu',kernel_initializer='normal')
#h3 = layers.Dense(256, activation = 'selu',kernel_initializer='normal')
#h4 = layers.Dense(256, activation = 'selu',kernel_initializer='normal')
h1 = layers.Dense(128, activation = 'selu',kernel_initializer='normal')
h2 = layers.Dense(64, activation = 'selu',kernel_initializer='normal')
h3 = layers.Dense(32, activation = 'selu',kernel_initializer='normal')
out_layer = layers.Dense(1) # linear activation because we are using continuous value. linear and no activation is one and the same. it is the default.


# x = normalize(inputs) - pairs with the linear normalization method. 
x = h1(inputs)
x = h2(x)
x = h3(x)
#x = h4(x)
#x = h5(x)
#x = h6(x)
#x = h7(x)
outputs = out_layer(x)

# Create the model by using the Model Object
model = keras.Model(inputs = inputs, outputs = outputs)
model.summary()

#%%
# 6. Compile the model 

model.compile(optimizer = 'adam', loss = 'MeanAbsoluteError', metrics=['mae'])

#%%
# 7. Tensorboard

base_log_path = r"C:\Users\muham\Documents\dekstop\AI Machine Learning\Deep Learning with Python\tb_logs"
log_path = os.path.join(base_log_path, 'Project 2', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = TensorBoard(log_dir=log_path)

#%%
# 8. perform model training

BATCH_SIZE = 32
EPOCHS = 100

history = model.fit(x_train,y_train, validation_data = (x_val,y_val), batch_size = BATCH_SIZE, epochs = EPOCHS, callbacks = [tb])
 
#%%
# 9. Visualize the result of model training.
 
training_loss = history.history["loss"]
val_loss = history.history['val_loss']
training_acc = history.history["mae"]
val_acc = history.history["val_mae"]
epochs_x_axis = history.epoch

plt.plot(epochs_x_axis, training_loss, label = "Training Loss")
plt.plot(epochs_x_axis, val_loss, label = 'Validation Loss')
plt.title("Training vs Validation Loss")
plt.legend()
plt.figure()

plt.plot(epochs_x_axis, training_acc, label = "Training MAE")
plt.plot(epochs_x_axis, val_acc, label = 'Validation MAE')
plt.title("Training vs Validation MAE")
plt.legend()
plt.figure()
plt.show()

#%%

#9. make predictions with your model

predictions = model.predict(x_test)

#%%

# 10. plot the predictions again labels 

theta = np.polyfit(y_test, predictions, 1)
y_line = theta[1] + theta[0] * y_test

plt.scatter(y_test, predictions)
plt.plot(y_test, y_line, 'r')
plt.title('Prediction Vs Labels')
plt.figure()
plt.show()

print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE :', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#%%
pred_vs_label = np.concatenate((predictions, np.expand_dims(y_test, axis = 1)), axis = 1)
