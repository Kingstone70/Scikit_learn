#!/usr/bin/env python
# coding: utf-8

# ## Algorithms  Comparison
# ####  Wilson adejo
# #### 21-12-2022

# In[2]:


pip install lazypredict


# In[3]:


import lazypredict
import pandas as pd


# In[4]:


from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# In[5]:


data = load_breast_cancer()
X = data.data
y= data.target


# In[6]:


X.shape


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)


# # Method 2

# In[8]:


data = pd.read_csv( "breast_cancer_data.csv")
data.head()


# In[9]:


# Splitting the loaded dataframe into Input(X) and Output(y) features
# Splitting into input features and assigning it a variable X
X = data[["radius_mean", "texture_mean", "perimeter_mean", "area_mean", 
        "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", 
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", 
        "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
        "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
        "symmetry_worst", "fractal_dimension_worst"]]

# Splitting into output feature and assigning it a label variable y 
y = data["diagnosis"]


# In[10]:


X.head()
#y.head()


# In[11]:


# Describing the head of the loaded dataframe 
data.describe()


# In[12]:


# Displaying the shape of the input and output dataset 
print('Input Shape: {}'.format(X.shape))
print('Output Shape: {}'.format(y.shape))


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)


# # Regression

# In[14]:


from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np

boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)

offset = int(X.shape[0] * 0.9)

X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)


# In[ ]:




