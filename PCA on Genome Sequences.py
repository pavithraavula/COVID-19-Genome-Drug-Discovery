#!/usr/bin/env python
# coding: utf-8

# ## Clustering and Visualization

# In[2]:


### Loading in packages

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns


# In[3]:


### Reading in data. Our data is stored in a dataframe called 'df'.

df = pd.read_csv('Alignment-HitTable.csv', header = None)
df.columns = ['query acc.verr', 'subject acc.ver', '% identity', 'alignment length', 'mismatches', 
             'gap opens', 'q. start', 'q. end', 's. start', 's. end', 'evalue', 'bit score']
df.head()


# In[4]:


## Part A (10 pts):

## Use the .describe() method to analyze the dataframe. Store the results in a new dataframe called 'results'.

# your code here
results = df.describe()

print(results)


# In[ ]:





# In[5]:


## Part B (10 pts):

## Store the correlation of the dataframe in a variable called 'corr'.

# your code here
corr = df.corr()
print(corr)


# In[6]:



fig, ax = plt.subplots(figsize = (10,10))
sns.heatmap(corr, annot = True, cmap = 'YlGnBu')
plt.show()


# In[7]:


## Part C (30 pts)

## Perform PCA with 2 components on the numeric columns of the datafame. Fit the PCA in a variable called 'pca'.


# your code here
df_numeric = df[['% identity', 'alignment length', 'mismatches', 'gap opens', 'q. start', 'q. end',
        's. start', 's. end', 'evalue', 'bit score']]

pca = PCA(n_components=2)
pca.fit(df_numeric);


# In[ ]:





# In[8]:


### Part D (25 pts)

## Store the components of the PCA in a dataframe called 'components'. Name the index as ['Component 1', 'Component 2'].

components = pd.DataFrame(pca.components_)

components.columns = ['% identity', 'alignment length', 'mismatches', 'gap opens', 'q. start', 'q. end',
        's. start', 's. end', 'evalue', 'bit score']

components.index = ['Component 1', 'Component 2']


# In[9]:




fig, ax = plt.subplots(figsize = (10,3))
sns.heatmap(components, annot = True, cmap = 'YlGnBu')
plt.show()


# In[10]:


### Part E (25 pts)

## Fit a K-Means clustering algorithm on the numeric data with 2 clusters and a random state of 0.
## Store the predicted groups in a variable called 'y_pred'

# your code here

# Fit K-Means with 2 clusters and a random state of 0
kmeans = KMeans(n_clusters=2, random_state=0).fit(df_numeric)
y_pred = kmeans.predict(df_numeric)

# Store the predicted groups in 'y_pred'
print(y_pred)


# In[11]:



sns.scatterplot(x= pca.transform(df_numeric)[:, 0], y = pca.transform(df_numeric)[:, 1], hue = y_pred)


# In[ ]:




