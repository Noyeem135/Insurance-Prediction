#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn import linear_model
import numpy as np


# In[10]:


df=pd.read_csv(r"E:\AI\insurance.csv")


# In[11]:


df.head(5)


# In[12]:


x=df.drop(['region'],axis=1)
x=x.dropna()
y = x['charges']
x = x.drop(['charges'],axis=1)


# In[13]:


x=x.dropna()


# In[14]:


x


# In[15]:


from sklearn.preprocessing import LabelEncoder


# In[16]:


le =LabelEncoder()


# In[17]:


x['sex']= le.fit_transform(df['sex'])


# In[18]:


x['sex'].unique()


# In[19]:


x


# In[20]:


x['smoker']= le.fit_transform(df['smoker'])


# In[21]:


x['smoker'].unique()


# In[22]:


x


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=79)


# In[35]:


x_train


# In[36]:


x_train.shape


# In[37]:


x_test.shape


# In[38]:


y_test.shape


# In[39]:


y_train.shape


# In[26]:


x_test


# In[27]:


y_train


# In[28]:


y_test


# In[40]:


from sklearn.linear_model import LinearRegression


# In[41]:


reg = LinearRegression()


# In[42]:


reg.fit(x_train, y_train)


# In[68]:


x


# In[43]:


reg.predict([[90,1,33.9000,2,1]])


# In[44]:


reg.score(x_test,y_test)


# In[45]:


reg.score(x_train,y_train)


# In[73]:


reg.coef_


# In[74]:


reg.intercept_


# In[46]:


from sklearn.linear_model import Ridge


# In[47]:


reg = Ridge()


# In[48]:


reg.fit(x_train, y_train)


# In[49]:


reg.predict([[90,1,33.9000,2,1]])


# In[50]:


reg.score(x_test,y_test)


# In[ ]:




