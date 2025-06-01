#!/usr/bin/env python
# coding: utf-8

# ## K-means clustering
# 
# In this notebook, we'll cluster sequences to find similar sequences with similar patterns.

# In[1]:


### Loading in libraries and packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[2]:


### Read in data. We'll store our data in a dataframe called 'df'

df = pd.read_csv('Alignment-HitTable.csv', header = None)
df.columns = ['query acc.verr', 'subject acc.ver', '% identity', 'alignment length', 'mismatches', 
             'gap opens', 'q. start', 'q. end', 's. start', 's. end', 'evalue', 'bit score']
df.head()


# In[4]:


## Part A (25 pts)

## Fit a K-means clustering with 5 clusters and a random state of 10 on the numeric columns in the dataframe.
## Store the predicted groups in a variable called 'y_pred'. 

# your code here
# Extracting the relevant numeric columns for clustering
numeric_columns = ['% identity', 'alignment length', 'mismatches', 'gap opens', 'q. start', 'q. end', 
                   's. start', 's. end', 'evalue', 'bit score']

# Dropping any rows with missing values in the numeric columns to ensure clean data for clustering
df_numeric = df[numeric_columns].dropna()

# Initializing KMeans with 5 clusters and random state of 10
kmeans = KMeans(n_clusters=5, random_state=10)

# Fitting the model to the data
y_pred = kmeans.fit_predict(df_numeric)


# In[ ]:





# In[6]:


### Part B (15 pts)

## Store the silhouette score on the predicted groups in a variable called 'score'.
## Hint: Use https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

# your code here

from sklearn.metrics import silhouette_score
# Calculate silhouette score
score = silhouette_score(df_numeric, y_pred, metric='euclidean')

# Output the silhouette score
score


# In[ ]:





# In[7]:


## Part C (30 pts)

## Store the silhouette scores for clusters 2 to 9 in a list called 'silhouette_scores'.
## Use a random state of 0 for each prediction.

# your code here

# Initialize an empty list to store silhouette scores
silhouette_scores = []

# Iterate over the range of clusters from 2 to 9
for n_clusters in range(2, 10):
    # Fit KMeans with the current number of clusters and random state 0
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    y_pred = kmeans.fit_predict(df_numeric)
    
    # Calculate the silhouette score for the predicted clusters
    score = silhouette_score(df_numeric, y_pred, metric='euclidean')
    
    # Append the score to the list
    silhouette_scores.append(score)

# Display the silhouette scores for clusters 2 to 9
silhouette_scores


# In[8]:


plt.bar(range(2, len(silhouette_scores) +2), silhouette_scores)
plt.show()


# In[9]:


## Part D (30 pts)

## Use a K-means clustering with 5 clusters on the normalized numeric dataframe. Use a random state of 0.
## Hint: Use the https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html to scale the data.
## Store the cluster centers in a dataframe called 'cluster_centers'. 
## Use the index ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'] on the dataframe.

# your code here
from sklearn.preprocessing import StandardScaler

# Standardize (normalize) the numeric data
scaler = StandardScaler()
df_numeric_scaled = scaler.fit_transform(df_numeric)

# Perform KMeans clustering with 5 clusters and random state 0
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(df_numeric_scaled)

# Get the cluster centers
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, 
                               columns=df_numeric.columns, 
                               index=['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'])

# Display the cluster centers
cluster_centers


# In[ ]:




