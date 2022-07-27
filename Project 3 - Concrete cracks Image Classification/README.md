# Image Classification - Concrete Cracks

## Summary
Classifying the image in 2 classes which is positive and negative

- Positive: There is no crack in the wall
- Negative: Existence of crack in the wall

## Data Preparation

Dataset: [Concrete Image Dataset](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

1. Dataset was split into Train, validation and test  with ratio of 70:24:6
2. Created Prefetch dataset as it is in Batch dataset format 
3. Data Augmentation (Flip & Rotation) was executed on the data

## Model Architecture

Model Architecture of Mobile_netV2 was used for this project by using transfer learning
Classificaton Layer was created as below. Activation function of Sigmoid was used in the output layer. 

![Classification layers]()

below shows the accuracy and loss of the model before training. 

![before_training]()

##Result

After training, The accuracy increased up to 99.83% with losses of 0.41%

![after_training]()
