#!/usr/bin/env python
# coding: utf-8

# # Data Science Task sheet
# Wilson Adejo
# 25-03-2023

# ### Load Packages

# In[11]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Import Data

# In[2]:


df = pd.read_csv('zillow.csv')
df.columns


# In[3]:


df.head()


# In[12]:


df.dtypes


# In[17]:


df.describe()


# ### Data Preparation and Modelling

# In[4]:


# pandas drop columns using list of column names
data = df.drop(['Index',' Year','Currentyear'], axis=1)
data.columns


# In[19]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[6]:


X = data.iloc[:,0:5]
y = data.iloc[:,5:6]
print(X.head())
print(y.head())


# ### Train/Test Split

# In[7]:


from sklearn.model_selection import train_test_split
# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=0)


# ### Regression

# In[8]:


# Create linear regression object
reg = LinearRegression()
# Train the model using the training sets
reg.fit(X,y) 
# Make predictions using the testing set
y_pred = reg.predict(X_test) 


# In[9]:


reg.score(X, y)


# In[10]:


reg.predict([[4010,5,3,32309,17]])


# In[20]:


# Find the R^2
print('The R-square is: ', reg.score(X, y))


# In[21]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# In[22]:


pipe=Pipeline(Input)
pipe


# In[23]:


pipe.fit(X,y)


# In[24]:


pipe.score(X,y)


# ### MODEL EVALUATION AND REFINEMENT
# import the necessary modules

# In[26]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
print("done")


# In[32]:


pr=PolynomialFeatures(degree=2)
X_train_pr=pr.fit_transform(X_train)
X_test_pr=pr.fit_transform(X_test)


# In[31]:


from sklearn.linear_model import Ridge


# In[33]:


RigeModel=Ridge(alpha=0.1)
RigeModel.fit(X_train_pr, y_train)
yhat = RigeModel.predict(X_test_pr)


# In[34]:


print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)


# In[35]:


RigeModel.score(X_train_pr, y_train)


# In[ ]:


poly = LinearRegression()
poly.fit(X_train_pr, y_train)


# In[37]:


poly.score(X_train_pr, y_train)


# In[38]:


poly.score(X_test_pr, y_test)


# In[ ]:


width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(alpha,Rsqu_test, label='validation data  ')
plt.plot(alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()


# In[ ]:




