#!/usr/bin/env python
# coding: utf-8

# ## Predicting bit scores
# 
# In this final notebook, we'll be predicting 'bit_score' from some of the columns in the data.

# In[1]:


### Loading in data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


### Reading in data. We'll store our data in dataframe called 'df'.

df = pd.read_csv('Alignment-HitTable.csv', header = None)
df.columns = ['query acc.verr', 'subject acc.ver', '% identity', 'alignment length', 'mismatches', 
             'gap opens', 'q. start', 'q. end', 's. start', 's. end', 'evalue', 'bit score']
df.head()


# In[3]:


## Part A (25 pts)

## In this final notebook, we'll be predicting 'bit_score' from some of the columns in the data.
## Create a feature dataframe called 'X' with the columns: ['% identity',  'mismatches', 'gap opens', 'q. start', 's. start'].
## Store the target 'y' with the bit scores.

# your code here
# Create the feature dataframe 'X' with selected columns
X = df[['% identity', 'mismatches', 'gap opens', 'q. start', 's. start']]

# Create the target 'y' with the bit scores
y = df['bit score']

# Display the feature dataframe and target to check the results
X.head(), y.head()


# In[ ]:





# In[5]:


## Part B (15 pts)

### Use the Standard Scaler from the sklearn.preprocessing library to normalize the data.
## Store the transformed X data in a variable called 'X_transformed'.

from sklearn.preprocessing import StandardScaler

# your code here

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the data using the StandardScaler
X_transformed = scaler.fit_transform(X)


# In[ ]:





# In[16]:


## Part C (25 pts)

## Predict the bit score by fitting a linear regression model. Store the predicted bit scores in
## a variable called 'lin_pred'. Get the score in a variable called 'lin_score'. 
## Store the linear regression coefficients in a variable called 'coef'.

# your code here

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

linreg = LinearRegression()
linreg.fit(X_transformed, y)
score = linreg.score(X_transformed, y)
lin_pred = linreg.predict(X_transformed)
coef = linreg.coef_


# In[ ]:





# In[17]:


from sklearn.neural_network import MLPRegressor

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, random_state = 42)
nn = MLPRegressor(random_state=42, solver = 'sgd', hidden_layer_sizes = (50,3)).fit(X_train, y_train)
nn_pred = nn.predict(X_test)
nn_score = nn.score(X_test, y_test)


# In[ ]:




