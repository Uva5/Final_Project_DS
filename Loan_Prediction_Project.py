#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression


# In[2]:


#Loading the data
df = pd.read_csv('Loan_Status_train.csv')


# In[3]:


#Printing the first 5 rows of the dataframe for sample view
df.head()


# In[4]:


# Finding the number of rows and columns and statistical information of the dataframe
df.info()
df.shape
df.describe()


# In[5]:


#Finding the missing values in the dataframe
df.isnull().sum()


# In[6]:


#Dropping all the missing values from the dataframe
df = df.dropna()


# In[7]:


#Finding the missing values in the dataframe
df.isnull().sum()


# In[8]:


#Label Encoding
df.replace({"Loan_Status":{"N":0, "Y":1}}, inplace=True)


# In[9]:


#Checking the label encoding
df.head()


# In[10]:


#Checking the Dependent column values
df["Dependents"].value_counts()


# In[11]:


df = df.replace({"Dependents":{"3+":4}})


# In[12]:


df["Dependents"].value_counts()


# In[13]:


#Visualizing this dataframe
# Education and Loan Status

sns.countplot(x="Education",hue="Loan_Status",data=df)


# In[14]:


#No.of loan approved for graduated people are more than that of not graduated people


# In[15]:


# Visualization for Marital Status and Loan Status

sns.countplot(x="Married", hue="Loan_Status", data=df)


# In[16]:


#No.of loan approved for married people are more than that of not married people


# In[17]:


# Visualization for Gender and Loan Status

sns.countplot(x="Gender", hue="Loan_Status",data=df)


# In[18]:


#No.of loan approved for Male is more than that of Female


# In[19]:


# Visualization for Self-employed and Loan Status

sns.countplot(x="Self_Employed", hue="Loan_Status", data=df)


# In[20]:


#No.of loan approved for self-employed is less 


# In[21]:


#converting categorical columns to numerical values

df.replace({"Gender":{"Male":1,"Female":0},
                 "Married":{"Yes":1,"No":0},
                 "Education":{"Graduate":1,"Not Graduate":0},
                 "Self_Employed":{"Yes":1,"No":1},
                 "Property_Area":{"Rural":0,"Semiurban":1,"Urban":2}
                },inplace=True)


# In[22]:


df.head()


# In[23]:


# Remove or drop not needed columns


# In[24]:


#Separating data and label

x = df.drop(columns=["Loan_ID","Loan_Status"], axis=1)
y = df["Loan_Status"]


# In[25]:


print(x)
print(y)


# In[26]:


#Splitting the data into train data and test data


# In[27]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=5)


# In[31]:


print(x.shape,x_train.shape,x_test.shape)


# In[32]:


#Training the Model: Logistic regression


# In[33]:


classifier = LogisticRegression()


# In[37]:


#Training the LogisticRegression Model

classifier.fit(x_train, y_train)


# In[36]:


#Model Evaluation


# In[38]:


from sklearn.metrics import accuracy_score


# In[39]:


#accuracy score on training data


# In[44]:


x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)


# In[45]:


print("Accuracy on Training Data: ",training_data_accuracy)


# In[46]:


#accuracy score on test data


# In[48]:


x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)


# In[49]:


print("Accuracy on Test Data: ",test_data_accuracy)


# In[ ]:




