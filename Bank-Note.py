#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("C:/Users/kushal/Desktop/Banknote-Authentcation/BankNote_Authentication.csv")


# In[3]:


df


# In[4]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[5]:


x.head()


# In[6]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[8]:


print(f'X_train shape is {x_train.shape}')
print(f'y_train shape is {y_train.shape}')

print(f'X_test shape is {x_test.shape}')
print(f'X_test shape is {y_test.shape}')


# In[9]:


scaler.fit(x_train)


# In[10]:


x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[ ]:





# In[11]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train_scaled,y_train)


# In[13]:


y_pred=classifier.predict(x_test_scaled)


# In[14]:


len(y_pred)


# In[15]:


from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test)


# In[16]:


score


# In[24]:


import pickle
pickle_out=open("classifier.pkl",'wb')
pickle.dump(classifier,pickle_out)
pickle_out.close()


# In[26]:


classifier.predict([[2,3,4,1],[2,3,11,5]])


# In[ ]:




