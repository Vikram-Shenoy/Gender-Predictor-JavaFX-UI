#!/usr/bin/env python
# coding: utf-8

# In[1]:


# EDA packages
import pandas as pd
import numpy as np
import requests
import io


# In[2]:


# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
#from sklearn.feature_extraction.text import TfidfVectorizer


# In[92]:



# Downloading the csv file from your GitHub account

url = "https://raw.githubusercontent.com/Jcharis/Python-Machine-Learning/master/Gender%20Classification%20With%20%20Machine%20Learning/names_dataset.csv" # Make sure the url is the raw version of the file on GitHub
download = requests.get(url).content

# Reading the downloaded content and turning it into a pandas dataframe

df = pd.read_csv(io.StringIO(download.decode('utf-8')),index_col=False)

# Printing out the first 5 rows of the dataframe

print (df.head())


# In[93]:


df.head()


# In[94]:


df.size


# In[95]:


# Data Cleaning
# Checking for column name consistency
df.columns


# In[96]:


# Data Types
df.dtypes


# In[97]:


# Checking for Missing Values
df.isnull().isnull().sum()


# In[98]:


# Number of Female Names
df[df.sex == 'F'].size


# In[99]:


# Number of Male Names
df[df.sex == 'M'].size


# In[100]:


df_names = df


# In[103]:


import seaborn as sns
sns.countplot(df_names['sex'], label = 'count')


# In[12]:


# Replacing All F and M with 0 and 1 respectively
df_names.sex.replace({'F':0,'M':1},inplace=True)


# In[13]:


df_names.sex.unique()


# In[14]:


df_names.dtypes


# In[15]:


Xfeatures =df_names['name']


# In[16]:


# Feature Extraction 
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)


# In[17]:


cv.get_feature_names()


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


# Features 
X
# Labels
y = df_names.sex


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[21]:


# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# In[22]:


# Accuracy of our Model
print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")


# In[23]:


# Accuracy of our Model
print("Accuracy of Model",clf.score(X_train,y_train)*100,"%")


# Therefore as training is 100% and test is 63% are model is Overfitting
# 

# In[24]:


# Sample1 Prediction
sample_name = ["Mary"]
vect = cv.transform(sample_name).toarray()


# In[25]:


vect


# In[26]:


# Female is 0, Male is 1
clf.predict(vect)


# In[27]:


# Sample2 Prediction
sample_name1 = ["Mark"]
vect1 = cv.transform(sample_name1).toarray()


# In[28]:


clf.predict(vect1)


# In[29]:


# Sample3 Prediction Names
sample_name2 = ["Natasha"]
vect2 = cv.transform(sample_name2).toarray()


# In[30]:


clf.predict(vect2)


# In[31]:


# Sample3 Prediction of Random Names
sample_name3 = ["Nefertiti","Nasha","Ama","Ayo","Xhavier","Ovetta","Tathiana","Xia","Joseph","Xianliang"]
vect3 = cv.transform(sample_name3).toarray()


# In[32]:


clf.predict(vect3)


# In[33]:


# A function to do it
def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")


# In[34]:


genderpredictor("Martha")


# In[35]:


namelist = ["Yaa","Yaw","Femi","Masha"]
for i in namelist:
    print(genderpredictor(i))


# <b>Using a custom function for feature analysis</b>
# 

# In[36]:


# By Analogy most female names ends in 'A' or 'E' or has the sound of 'A'
def features(name):
    name = name.lower()
    return {
        'first-letter': name[0], # First letter
        'first2-letters': name[0:2], # First 2 letters
        'first3-letters': name[0:3], # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }


# In[37]:


# Vectorize the features function
features = np.vectorize(features)
print(features(["Anna", "Hannah", "Peter","John","Vladmir","Mohammed"]))


# In[38]:


# Extract the features for the dataset
df_X = features(df_names['name'])


# In[39]:


df_y = df_names['sex']


# In[40]:


from sklearn.feature_extraction import DictVectorizer
 
corpus = features(["Mike", "Julia"])
dv = DictVectorizer()
dv.fit(corpus)
transformed = dv.transform(corpus)
print(transformed)


# In[41]:


dv.get_feature_names()


# In[42]:


# Train Test Split
dfX_train, dfX_test, dfy_train, dfy_test = train_test_split(df_X, df_y, test_size=0.33, random_state=42)


# In[43]:


dfX_train


# In[44]:


dv = DictVectorizer()
dv.fit_transform(dfX_train)


# In[45]:


# Model building Using DecisionTree

from sklearn.tree import DecisionTreeClassifier
 
dclf = DecisionTreeClassifier()
my_xfeatures =dv.transform(dfX_train)
dclf.fit(my_xfeatures, dfy_train)


# In[46]:


# Build Features and Transform them
sample_name_eg = ["Alex"]
transform_dv =dv.transform(features(sample_name_eg))


# In[47]:


vect3 = transform_dv.toarray()


# In[48]:


# Predicting Gender of Name
# Male is 1,female = 0
dclf.predict(vect3)


# In[49]:


if dclf.predict(vect3) == 0:
    print("Female")
else:
    print("Male")


# In[50]:


# Second Prediction With Nigerian Name
name_eg1 = ["Chioma"]
transform_dv =dv.transform(features(name_eg1))
vect4 = transform_dv.toarray()
if dclf.predict(vect4) == 0:
    print("Female")
else:
    print("Male")


# In[51]:


# A function to do it
def genderpredictor1(a):
    test_name1 = [a]
    transform_dv =dv.transform(features(test_name1))
    vector = transform_dv.toarray()
    if dclf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")


# In[52]:


random_name_list = ["Alex","Alice","Chioma","Vitalic","Clairese","Chan"]


# In[53]:


for n in random_name_list:
    print(genderpredictor1(n))


# In[54]:


## Accuracy of Models Decision Tree Classifier Works better than Naive Bayes
# Accuracy on training set
print(dclf.score(dv.transform(dfX_train), dfy_train))


# In[55]:


# Accuracy on test set
print(dclf.score(dv.transform(dfX_test), dfy_test))


# We have reduced overfitting of the code using a decision tree classifier 
# # Plotting our results

# In[56]:


from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree


# In[112]:


prob=dclf.predict(dv.transform(dfX_test))#probable values


# In[90]:


#plotting our values
from sklearn.metrics import confusion_matrix


# In[113]:


confusion_matrix_graph = confusion_matrix(dfy_test,prob)
confusion_matrix_graph


# In[117]:


# IMPORTANT: first argument is true values, second argument is predicted values
# this produces a 2x2 numpy array (matrix)
print(confusion_matrix(dfy_test, prob))


# In[119]:


# print the first 25 true and predicted responses
print('True', dfy_test.values[0:25])
print('Pred', prob[0:25])


# In[122]:


# save confusion matrix and slice into four pieces
confusion = confusion_matrix(dfy_test, prob)
print(confusion)
#[row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]


# In[123]:


from sklearn.metrics import plot_confusion_matrix


# In[130]:


plot_confusion_matrix(dclf,dv.transform(dfX_test),dfy_test)
plt.savefig('Confusion.png')


# In[134]:


from sklearn import metrics


# In[140]:


y_pred_prob = dclf.predict_proba(dv.transform(dfX_test))[:, 1]


# In[151]:


# IMPORTANT: first argument is true values, second argument is predicted probabilities

# we pass dfy_test and prob
# we do not use y_pred_class, because it will give incorrect results without generating an error
# roc_curve returns 3 objects fpr, tpr, thresholds
# fpr: false positive rate
# tpr: true positive rate
fpr, tpr, thresholds = metrics.roc_curve(dfy_test, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for gender classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.savefig('ROC.png')


# In[152]:


#AUC is the percentage of the ROC plot that is underneath the curve:
# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(dfy_test, y_pred_prob))


# *SAVING OUR MODEL TO BE USED FOR JAVA FX*

# In[154]:


#Model Saving
import pickle
dctreeModel = open("namesdetectormodel.pkl","wb")
pickle.dump(dclf,dctreeModel)
dctreeModel.close()


# In[ ]:





# In[155]:


# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(dclf, open(filename, 'wb'))


# In[156]:


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(dv.transform(dfX_test), dfy_test)


# In[157]:


result


# In[ ]:




