#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The first step is to import the libraries necessary to perform the mathematical operations and the keras data set 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from keras.datasets import mnist 

# this loads the data from the mnist data set 
(xtrain, ytrain), (xtest,xtest) = mnist.load_data()

#this selects a subset of the mnist data (150 images)
xtrain=xtrain[:150]
y=ytrain[:150]

X=xtrain.T
X=X/255

y.resize((150,1))
y=y.T

#check the values 
pd.Series(y[0]).value_counts()

