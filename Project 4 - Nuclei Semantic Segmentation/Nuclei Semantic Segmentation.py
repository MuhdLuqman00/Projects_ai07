# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:39:01 2022

@author: muham
"""

# 0. Import packages 


import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, losses, optimizers, callbacks
#import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix # allow us to construct the expanding path 
from sklearn.model_selection import train_test_split 
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cv2, os, glob
import numpy as np
from scipy import io

#%%

# 1. Load the data

# 1.1  prepare empty list for images and masks
images_train = []
masks_train = []

images_test = []
masks_test = []
train_path = r"C:\Users\muham\Documents\dekstop\AI Machine Learning\Deep Learning with Python\Project 4 - Cell Nuclei Semantice Segmentation\data-science-bowl-2018-2\train"

test_path = r"C:\Users\muham\Documents\dekstop\AI Machine Learning\Deep Learning with Python\Project 4 - Cell Nuclei Semantice Segmentation\data-science-bowl-2018-2\test"

# 1.2 Load the images using opencv

image_train_dir = os.path.join(train_path, 'inputs')
for image_file in os.listdir(image_train_dir):
    img = cv2.imread(os.path.join(image_train_dir,image_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    images_train.append(img)
    
# 1.3 Load the masks
masks_train_dir = os.path.join(train_path, 'masks')
for mask_file in os.listdir(masks_train_dir):
    mask = cv2.imread(os.path.join(masks_train_dir,mask_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    masks_train.append(mask)
    

# 1.4 load images and masks for test 

image_test_dir = os.path.join(test_path, 'inputs')
for image_file in os.listdir(image_test_dir):
    img = cv2.imread(os.path.join(image_test_dir,image_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(128,128))
    images_test.append(img)
    
# 1.3 Load the masks
masks_test_dir = os.path.join(test_path, 'masks')
for mask_file in os.listdir(masks_test_dir):
    mask = cv2.imread(os.path.join(masks_test_dir,mask_file), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask,(128,128))
    masks_test.append(mask)
    



#%%
# 1.4 Convert the lists into numpy array 

images_train_np = np.array(images_train)
masks_train_np = np.array(masks_train)

images_test_np = np.array(images_test)
masks_test_np = np.array(masks_test)
#%%
# 1.5 Check some examples train
plt.figure(figsize = (10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(images_train[i])
    plt.axis('off')
    
plt.show()

plt.figure(figsize = (10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(masks_train_np[i])
    plt.axis('off')
    
plt.show()

# 1.6 Check some examples test
plt.figure(figsize = (10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(images_test[i])
    plt.axis('off')
    
plt.show()

plt.figure(figsize = (10,10))
for i in range(1,4):
    plt.subplot(1,3,i)
    plt.imshow(masks_test_np[i])
    plt.axis('off')
    
plt.show()
#%%

# 2. Data Preprocessing
# 2.1 Expand the mask dimension
masks_train_np_exp = np.expand_dims(masks_train_np, axis = -1) # we want to add the channel axis 
masks_test_np_exp = np.expand_dims(masks_test_np, axis = -1)

# Check the mask output
print(np.unique(masks_train[0]))
print(np.unique(masks_test[0]))

#%%

# 2.2 Convert the mask values into class labels 

converted_train_masks = np.round(masks_train_np_exp/255)
converted_test_masks = np.round(masks_test_np_exp/255)

# Check the mask output
print(np.unique(converted_train_masks[0]))
print(np.unique(converted_train_masks[0]))

#%%

# 2.3 Normalize image pixels values

converted_train_images = images_train_np /255.0
sample = converted_train_images[0]

converted_test_images = images_test_np/255.0
sample = converted_test_images[0]

#%%
# 2.4 Convert the numpy arrays into tensor
x_train_tensor = tf.data.Dataset.from_tensor_slices(converted_train_images)
x_test_tensor = tf.data.Dataset.from_tensor_slices(converted_test_images)
y_train_tensor = tf.data.Dataset.from_tensor_slices(converted_train_masks)
y_test_tensor = tf.data.Dataset.from_tensor_slices(converted_test_masks)

#%%

# 2.5 Combine the images and masks using zip 

train_dataset = tf.data.Dataset.zip((x_train_tensor, y_train_tensor))
test_dataset = tf.data.Dataset.zip((x_test_tensor,y_test_tensor))

#%% 
# 2.6 Create a subclass layer for data augmentation 
class Augment(layers.Layer):
    def __init__(self,seed = 42):
        super().__init__()
        self.augment_inputs = layers.RandomFlip(mode = 'horizontal', seed = seed)
        self.augment_labels = layers.RandomFlip(mode = 'horizontal', seed = seed)
        
    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels 
    
#%%
# 2.7 Convert into prefetch dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_dataset)
STEPS_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE 

#%%

# 3. Build the input dataset 

train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size = tf.data.AUTOTUNE)
    )

test_batches = test_dataset.batch(BATCH_SIZE)

#%%


# 4. Visualize some examples 

def display(display_list):
    plt.figure(figsize = (15,15))
    title = ['Input Image', 'True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()
    
#%%
# 5. Examples
for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image,sample_mask])

#%%

# 6. Create the imagee segmentation model 

# 6.1 Use a pretrained model as the feature extraction layers 

base_model = keras.applications.MobileNetV2(input_shape = [128,128,3], include_top= False)

# 6.2 List down some activation layers 

layer_names = [
    'block_1_expand_relu', #64x64
    'block_3_expand_relu', #32x32
    'block_6_expand_relu', #16x16
    'block_13_expand_relu', #8x8
    'block_16_expand_relu', #4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

# Define the feature Extraction model 
down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)    

#%%

# Make use of the function to construct the entire U-net 

OUTPUT_CLASSES = 2
model = unet_model(output_channels=OUTPUT_CLASSES)

#compile the model 
loss = losses.SparseCategoricalCrossentropy(from_logits = True)

model.compile(optimizer = 'adam', loss = loss, metrics = ['accuracy'])
#%%
keras.utils.plot_model(model, show_shapes = True)

#%%

#Create function to show predictions 

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis = -1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset = None, num = 1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
            
    else:
        display([sample_image,sample_mask, create_mask(model.predict(sample_image[tf.newaxis,...]))])
        
#%%

# Test out the show_prediction function 
show_predictions()

#%%

# Create a callback to help display results during model training 

class DisplayCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        clear_output(wait = True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))


#%%

# 10. Model training 
# Hyperparameters for the model 

EPOCHS = 19
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(test_dataset)//BATCH_SIZE//VAL_SUBSPLITS

history = model.fit(train_batches, validation_data = test_batches, epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH, validation_steps = VALIDATION_STEPS, callbacks = [DisplayCallback()])

#%%

# 11. Deploy Model 

show_predictions(test_batches, 3)
