# Image-Classification

*Aim*: The aim of this project is to create, train and evaluate a neural network in TensorFlow, understand the basics of neural networks and solve classification problems with neural networks.

## Theoretical Overview

*TensorFlow:* TensorFlow is a free open source framework which enables users to develop end-to-end machine learning and deep learning and deep learning projects, starting from pre-processing to model training and deployment. It is initially developed by the *Google Brain* team for internal use within *Google*, but now it's useage has been widespread.

*Keras:* Keras is an open-source neural-network Python library, capable of running on top of TensorFlow. It is designed to enable fast experimentation with deep neural networks.

*One Hot Encoding:* It refers to splitting the column which contains numerical categorical data to many columns depending on the number of categories present in that column. Each column contains “0” or “1” corresponding to which column it has been placed.

## For Data Modelling  
       import numpy as np
                    
## For Data Visualization
       import matplotlib.pyplot as plt
                 
## For Neural Network 
       import tensorflow as tf 

## For Importing Dataset
       from tensorflow.keras.datasets import mnist
                    
## For One Hot Encoding
       from tensorflow.python.keras.utils import to_categorical
                    
## For Creating the Model
       from tensorflow.keras.models import Sequential
       from tensorflow.keras.layers import Dense

## Name of the dataset : 
       MNIST
                    
## Tasks:
       Loading the data
       Perform One Hot Encoding
       Preprocessing
       Create model
       Train the model
       Evaluate the model
       Predictions

## For Data Visualization, I used:
       line plot
