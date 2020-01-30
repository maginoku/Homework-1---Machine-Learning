#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np #importing important packages
import pandas as pd #importing important packages
import copy
from sklearn import preprocessing #this will help in making the logistic regression.
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split #import module to split dataset
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn import tree


# In[46]:


train = pd.read_csv(r"C:\Users\Nathan Campbell\Documents\Machine Leaning\Homeworks\Homework 1\Titanic\train.csv") 
#importing the train.csv file

test = pd.read_csv(r"C:\Users\Nathan Campbell\Documents\Machine Leaning\Homeworks\Homework 1\Titanic\test.csv") 
#importing the test.csv file


# In[47]:


train.head() #you can see the features of the data set here


# In[48]:


train.describe() #looking at the table to do further analysis, helps you find the stats infor like: Mean, SD, min, Max, etc


# In[49]:


train.isnull().sum() #This checks how many missing values we have


# In[50]:


train.info()


# In[51]:


train.shape


# In[52]:


train = train.fillna({"Embarked": "S"})
test = test.fillna({"Embarked": "S"})


gender_mapping = {"male": 1, "female": 0}
train['Sex'] = train['Sex'].map(gender_mapping)
test['Sex'] = test['Sex'].map(gender_mapping)

Embarked_mapping = {"S":1, "C":2, "Q":3}
train['Embarked'] = train['Embarked'].map(Embarked_mapping)
test['Embarked'] = test['Embarked'].map(Embarked_mapping)


# In[55]:


#Since children 18 and younger were taken off the ship first, there should be a division.

for x in range(len(train["Age"])):
    if train["Age"][x] < 18:
        train["Age"][x] = 1

    if 18 <= train["Age"][x] < 9999:
        train["Age"][x] = 2
        
train["Age"] = train["Age"].fillna(1)

for x in range(len(test_data["Age"])):
    if test["Age"][x] < 18:
        test["Age"][x] = 1
        
    if 18 <= test["Age"][x] < 9999:
        test["Age"][x] = 2
        
test["Age"] = test["Age"].fillna(1)        


# In[37]:


#Here I am splitting my training data to test the accuracy of my training model to decide whether Logistic Regression  is a fairly accurate model to use
predictors = train.drop(['Survived', 'PassengerId','Cabin','Ticket','Name'], axis=1)
response = train["Survived"]
x_train_value, x_value, y_train_value, y_value = train_test_split(predictors, response, test_size = 0.20, random_state = 0)


# In[35]:


logreg = LogisticRegression()
logreg.fit(x_train_value, y_train_value)
y_predict = logreg.predict(x_value)
accuracy_logreg = round(accuracy_score(y_predict, y_value) * 100, 2)
print(accuracy_logreg)


# In[ ]:


#Dropping the features that I dont want to use.
test = test.drop(["Cabin",'Ticket','Name'], axis = 1)


# In[ ]:


#Test the test data

test["Fare"] = test["Fare"].fillna(1)

LogNog = logreg.predict(test.drop(['PassengerId'],axis = 1))

PassID = test["PassengerId"]

output = pd.DataFrame({ 'PassengerId' : PassID, 'Survived': LogNog })
output.to_csv('Results(HW1).csv', index=False)

