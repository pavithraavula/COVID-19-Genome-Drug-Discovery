#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Importing packages and reading in data.
### Genome sequencing data is stored in the dataframe pn.
### Notice how each column has separate sequencing data.
### Our goal will be to find similarities between the sequences and characterize it.

import pandas as pd
import numpy as np

pn = pd.read_csv('SARS_CORONAVIRUS_NC_045512_sequence.fasta', header = None)[1:][0]
pn = pd.DataFrame(pn)
pn.columns = ['Genome Sequence']


# In[ ]:


## Part A (20 pts):

## How many letters are in each sequence? Find the length of the sequence in each row and store the results in a new
## column called ['Length']. Then, find the median length of these genome sequences and store it 
## in a variable called 'median_len'. Delete all rows that are not of this length.

pn['Length'] = pn['Genome Sequence'].apply(lambda x: len(x))
mid_length = np.median(pn['Length'])
pn = pn[pn.Length == mid_length]


# In[ ]:


## Part B (20 pts):

## Create a new column titled 'Match?' with a 1/0 corresponding to whether or not the sequence
## contains the substring 'TAATTTAGGCATGCCTT'.

pn['Match?'] = pn['Genome Sequence'].apply(lambda x: x.find('TAATTTAGGCATGCCTT'))
pn['Match?'] = pn['Match?'].apply(lambda x: 0 if x == -1 else 1)


# In[ ]:


## Part C (35 pts):

## Create a matrix called 'differences' storing the number of differences in the sequences between the rows.

def diff_letters(a,b):
    return sum ( a[i] != b[i] for i in range(len(a)) )

differences = []
for name, row in pn.iterrows():
    temp_diff = []
    for j in range(len(pn)):
        temp_diff.append(diff_letters(row[0], pn.iloc[j]['Genome Sequence']))
    differences.append(temp_diff) 


# In[ ]:


## Part D (25 pts):

## Find the two genome sequences that are most similar. Store them in a tuple called 'similar' with the format (row1, row2).

min_val = 1000
similar = 0
for i in range(len(differences)):
    for j in range(len(differences[0])):
        if differences[i][j] < min_val and differences[i][j] != 0:
            min_val = differences[i][j]
            similar = (i,j)

