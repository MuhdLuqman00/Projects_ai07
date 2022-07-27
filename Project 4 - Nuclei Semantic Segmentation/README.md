# Nuclei Image Segmentation

## Summary 

Creating a semantic segmentation model that detects nuclei in images

## Dataset 

Link: [Nuclei Semantic Segmentation Dataset](https://www.kaggle.com/competitions/data-science-bowl-2018/overview)

1. Loading Train dataset (Images and Masks) and Test Dataset (Images and Masks)

![Train images sample](images/)

2. Change the data type to Numpy arrays 
3. Expanding the dimension of Masks (train and test) to match with the dimension of the images's (train and test) dimension
4. Convert Masks Values into labels of [0. 1.]
5. Normalize image pixel values 
6. Combine Image and Masks in categories of Train and Test dataset
7. Build Input Dataset
