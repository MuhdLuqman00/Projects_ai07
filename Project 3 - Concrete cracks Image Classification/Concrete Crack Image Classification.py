# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:46:42 2022

@author: muham
"""

# 0.import packages 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, callbacks, applications
import numpy as np 
import os , datetime, pathlib
import matplotlib.pyplot as plt

#%%
# 1. Load dataset

file_path = r"C:\Users\muham\Documents\dekstop\AI Machine Learning\Deep Learning with Python\Project 3 - Classifying Concretes (Cracks)\Dataset"

dataset_dir = pathlib.Path(file_path)

SEED = 12345
IMG_SIZE = (160,160)
BATCH_SIZE = 16
train_dataset = keras.utils.image_dataset_from_directory(dataset_dir, seed = SEED, image_size = (160,160), batch_size = BATCH_SIZE, subset = 'training', validation_split = 0.3)
val_data = keras.utils.image_dataset_from_directory(dataset_dir, seed = SEED, image_size = (160,160), batch_size = BATCH_SIZE, subset = 'validation', validation_split = 0.3)

#%%
# 2. Split dataset 
# Split the validation dataset into validation and test datasets 

val_batches = tf.data.experimental.cardinality(val_data)
test_dataset = val_data.take(val_batches//5)
validation_dataset = val_data.skip(val_batches//5)

#%%
# 3. Prefetch the dataset because the dataset is in BatchDataset. 

AUTOTUNE = tf.data.AUTOTUNE
pf_train = train_dataset.prefetch(buffer_size = AUTOTUNE)
pf_val = validation_dataset.prefetch(buffer_size = AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size = AUTOTUNE)

#%%

# 4. Data Augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#%%
# 5. Prepare the deep learning model by applying transfer learning 
#       Before that, we want to create a layer to normalize the data. 

preprocess_input = applications.mobilenet_v2.preprocess_input
#preprocess_input = applications.VGG16.preprocess_input (slower than mobilenet_v2)

# We are using MobileNetV2 as the feature extractor 

IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = False)

#%%

# 6. We want to make base_model frozen ( not receiving training)

base_model.trainable = False 
base_model.summary()

#%%
# 7. We are building the classification layers here

global_avg = layers.GlobalAveragePooling2D()

# Add an output layer  or put more dense layer but GA is already good enough of a dense layer
class_names = train_dataset.class_names
output_layer = layers.Dense(len(class_names), activation = 'sigmoid')

#%%
# 8. Use functional API to like all the layers together 

inputs = keras.Input(shape = IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x)
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs = inputs, outputs = outputs )
model.summary()

#%%
# 9. Compile the model 

optimizer = optimizers.Adam(learning_rate = 0.0001)

'''
You can also apply lerning rate schedule 

lr_schedule = optimizers.schedules.CosineDecay(0.001,500)
optimizer = optimizers.Adam(learning_rate =lr_schedule)
'''


loss = losses.SparseCategoricalCrossentropy()

model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])

#%%
# 10. Evaluate the model before model training 

loss0, accuracy0 = model.evaluate(pf_val)

print("------------------------------------Before Training--------------------------------")
print('Loss = ', loss0)
print('accuracy =', accuracy0)

#%%
# 11.TensorBoard Callback

base_log_path = r"C:\Users\muham\Documents\dekstop\AI Machine Learning\Deep Learning with Python\tb_logs"
log_path = os.path.join(base_log_path, 'Project 3', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_dir=log_path)

#%%

# 12.Train the model ( PNGCrush can be used to solve the "PNG warning: iCCP: known incorrect sRGB profile")

EPOCHS = 10

history = model.fit(pf_train, validation_data = pf_val, epochs = EPOCHS, callbacks = [tb])

#%%
# 13.Evaluate the model after training 

test_loss, test_accuracy = model.evaluate(pf_test)

print("-----------------------------------------------After Training--------------------------------")
print("Loss = ", test_loss)
print("Accuracy = ", test_accuracy)

#%%
# 14. Deploying the model
image_batch, label_batch = pf_test.as_numpy_iterator().next()
predictions = np.argmax(model.predict(image_batch), axis = 1)

# Compare label vs prediction 

label_vs_predictions = np.transpose(np.vstack((label_batch, predictions)))