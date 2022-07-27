# Nuclei Image Segmentation (U-Net)

## Summary 

Creating a semantic segmentation model that detects nuclei in images

## Dataset 

Link: [Nuclei Semantic Segmentation Dataset](https://www.kaggle.com/competitions/data-science-bowl-2018/overview)

Dataset are loaded and undergoes preprocessing before it can be applied in the U-net semantic segmentation model. preprocessing have the following order.

1. **Images Dataset** --> **Numpy Arrays** --> **Normalization** --> **Tensor** --> **Prefetch**

2. **Masks Dataset** --> **Numpy Arrays** --> **Dimension Expansion** --> **Mask Labels** --> **Tensor** --> **Prefetch**


Below are some examples of images and masks dataset

![Train images sample](images/)
![Train masks sample](images/)
![Test images sample](images/)
![Test masks sample](images/)

## Before Model Training

As you can see from the image below, the model did not do a very good job in predicting the masks of the images 

![Train images sample](images/)

## After Model Training

The model was trained for 19 epoch which resulted in the accuracy of 96.96%  with the loss of 7.45 percent

![Train images sample](images/)

