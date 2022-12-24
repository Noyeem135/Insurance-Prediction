#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import linear_model
import numpy as np


# In[2]:


df = pd.read_csv('insurance.csv')


# In[3]:


df.head(5)


# In[4]:


df.shape


# In[5]:


x = df.drop(['region'], axis=1)
x = x.dropna()

y =x['charges']
x = x.drop(['charges'], axis=1)
#y =x['charges']


# In[6]:


x = x.dropna()


# In[7]:


x


# In[67]:


x.shape


# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


le = LabelEncoder()


# In[11]:


x['sex'] = le.fit_transform(df['sex'])


# In[12]:


x['sex'].unique()


# In[13]:


x['smoker'] = le.fit_transform(df['smoker'])


# In[14]:


x['smoker'].unique()


# In[15]:


x.head(5)


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=79)


# In[18]:


x_train


# In[19]:


x_train.shape


# In[20]:


x_test


# In[21]:


x_test.shape


# In[22]:


y_train


# In[23]:


y_train.shape


# # Linear Regression.

# In[24]:


from sklearn.linear_model import LinearRegression


# In[25]:


reg = LinearRegression()


# In[26]:


reg.fit(x_train, y_train)


# In[27]:


x.head(2)


# In[28]:


reg.predict([[28,1,27.1,0,1]])


# In[29]:


reg.score(x_test, y_test)


# In[30]:


reg.coef_


# In[31]:


reg.intercept_


# # Ridge Regression.

# In[59]:


from sklearn.linear_model import Ridge


# In[60]:


reg = Ridge()


# In[61]:


reg.fit(x_train, y_train)


# In[62]:


reg.fit(x_train, y_train)([[28,1,27.1,0,1]])


# In[63]:


reg.predict([[28,1,27.1,0,1]])


# In[64]:


#seo.
reg.coef_


# In[65]:


#seo.
reg.intercept_


# # Neural Network Regression.

# In[37]:


from sklearn.neural_network import MLPRegressor


# In[38]:


# note: reg Vs regr
regr = MLPRegressor


# In[39]:


# (random_state=1, max_iter=500) [why use this extra]
regr = MLPRegressor(random_state=1, max_iter=500).fit(x_train, y_train)


# In[40]:


regr.predict([[28,1,27.1,0,1]])


# In[ ]:


#seo.
reg.coef_
#MLPRegressor.coef_


# In[ ]:


#seo.
reg.intercept_


# # Lasso Regression.

# In[41]:


from sklearn import linear_model


# In[42]:


# what is the uses of alpha..?? clf..??

clf = linear_model.Lasso(alpha=0.1)


# In[43]:


clf.fit(x_train, y_train)


# In[44]:


clf.predict([[28,1,27.1,0,1]])


# # Decision Tree Regression.

# In[45]:


from sklearn.tree import DecisionTreeRegressor


# In[46]:


regressor = DecisionTreeRegressor(random_state=0)


# In[47]:


regressor.fit(x_train, y_train)


# In[48]:


regressor.predict([[28,1,27.1,0,1]])


# # Random Forest.

# In[49]:


from sklearn.ensemble import RandomForestRegressor


# In[50]:


# note: (max_depth=2, random_state=0)
regr = RandomForestRegressor(max_depth=2, random_state=0)


# In[51]:


regr.fit(x_train, y_train)


# In[52]:


regr.predict([[28,1,27.1,0,1]])


# # KNN Model.

# In[53]:


from sklearn.neighbors import KNeighborsRegressor


# In[54]:


neigh = KNeighborsRegressor(n_neighbors=2)


# In[55]:


neigh.fit(x_train, y_train)


# In[56]:


neigh.predict([[28,1,27.1,0,1]])


# In[57]:


neigh.score(x_test, y_test)


# # Support Vector Machines (SVM)

# In[68]:


from sklearn.svm import SVR


# In[69]:


regr = SVR(C=1.0, epsilon=0.2)


# In[70]:


regr.fit(x_train, y_train)


# In[71]:


regr.predict([[28,1,27.1,0,1]])


# In[ ]:





# In[ ]:





# In[ ]:




