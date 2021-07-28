#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pickle
# import pandas as pd
import numpy as np

# import requests
# import io

# In[106]:
# import sklearn
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction import DictVectorizer
# import sklearn.feature_extraction.text
# import sklearn.feature_extraction

# In[107]:


filename = 'C:\\Users\\Vikram\\Documents\\Interprocess\\finalized_model.sav'
filename1 = 'C:\\Users\\Vikram\\Documents\\Interprocess\\finalized_model1.sav'

# In[108]:


dclf = pickle.load(open(filename, 'rb'))
dv = pickle.load(open(filename1, 'rb'))


# In[109]:


# By Analogy most female names ends in 'A' or 'E' or has the sound of 'A'
def features(name):
    name = name.lower()
    return {
        'first-letter': name[0],  # First letter
        'first2-letters': name[0:2],  # First 2 letters
        'first3-letters': name[0:3],  # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }


# In[110]:


# Vectorize the features function
features = np.vectorize(features)


# In[111]:


def genderpredictor1(a):
    test_name1 = [a]
    transform_dv = dv.transform(features(test_name1))
    vector = transform_dv.toarray()
    fw = open("C:\\Users\\Vikram\\Documents\\SecondTry\\src\\sample\\results.txt", "w")
    if dclf.predict(vector) == 0:
        fw.write("Female")
    else:
        fw.write("Male")
    return


# print("Male")

# In[112]:


import sys

# In[113]:


# print("sys.argv is:", sys.argv)

# In[114]:


# a = "erika"
# genderpredictor1(a)
# print("String passed is "+a)
# print(pickle.format_version)
# print(sklearn.__version__)


# In[115]:
#
with open("C:\\Users\\Vikram\\Documents\\SecondTry\\name.txt") as f:
    a = f.read()
genderpredictor1(a)

# In[ ]:


# In[ ]:
