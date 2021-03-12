#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Importing important libraries 


# In[5]:


import numpy as np
import pandas as pd 


import matplotlib as mpl
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing 
from matplotlib.animation import FuncAnimation
from sklearn.datasets import load_boston 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from IPython.display import HTML


# # Loading the BOSTON HOUSING DATASET

# In[9]:


boston=load_boston()
#description of datasets 
print(boston.DESCR)


# In[10]:


#putting the data in panda dataframes
features=pd.DataFrame(boston.data ,columns=boston.feature_names)
features


# # Preprocessing 

# In[11]:


standardscaler=preprocessing.StandardScaler()
features_scaled=standardscaler.fit_transform(features)
features_scaled


# In[12]:


target=pd.DataFrame(boston.target,columns=['target'])
target


# In[13]:


df=pd.concat([features,target],axis=1)
df


# # visualization

# In[14]:


corr=df.corr('pearson')
corr=df.corr('pearson')
corrs=[abs(corr[attr]['target'])for attr in list(features)]
l=list(zip(corrs,list(features)))
l.sort(key=lambda x : x[0],reverse=True)
corrs,labels=list(zip((*l)))
index=np.arange(len(labels))
plt.figure(figsize=(15,5))
plt.bar(index,corrs,width=0.5)
plt.xlabel('Attributes')
plt.ylabel('correlation with the target variables ')
plt.xticks(index,labels)
plt.show()


# # Splitting the data

# In[15]:


X=df['LSTAT'].values
Y=df['target'].values


# In[16]:


print(Y[:5])


# In[17]:


x_scaler=MinMaxScaler()
X=x_scaler.fit_transform(X.reshape(-1,1))
X=X[:,-1]
y_scaler=MinMaxScaler()
Y=y_scaler.fit_transform(Y.reshape(-1,1))
Y=Y[:,-1]


# In[18]:


print(Y[:5])


# In[19]:


xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)
xtrain=xtrain.reshape(-1,1)
ytrain=ytrain.reshape(-1,1)
xtest=xtest.reshape(-1,1)
ytest=ytest.reshape(-1,1)


# # Training the model

# In[20]:


lm=LinearRegression()


# In[21]:


lm.fit(xtrain,ytrain)


# In[22]:


lm.coef_


# # Making predictions  

# In[23]:


predictions=lm.predict(xtest)


# In[24]:


plt.scatter(ytest,predictions)
plt.xlabel('ytest')
plt.ylabel('Prediction')


# In[25]:


from sklearn import metrics 
print('MSE:',metrics.mean_squared_error(ytest,predictions))
print('RMSE:',np.sqrt(metrics.mean_squared_error(ytest,predictions)))


# In[26]:


p=pd.DataFrame(list(zip(xtest,ytest,predictions)),columns=['x','target_y','predictions'])
p


# # Plotting the predicted values against the target values

# In[27]:


plt.scatter(xtest,ytest,color='b')
plt.plot(xtest,predictions,color='r')


# # Reverting normalization to obtain predicted prices of house

# In[28]:


predictions=np.array(predictions).reshape(-1,1)
xtest=xtest.reshape(-1,1)
ytest=ytest.reshape(-1,1)


xtest_scaled=x_scaler.inverse_transform(xtest)
ytest_scaled=y_scaler.inverse_transform(ytest)
predictions_scaled=y_scaler.inverse_transform(predictions)


xtest_scaled=xtest_scaled[:,-1]
ytest_scaled=ytest_scaled[:,-1]
predictions_scaled=predictions_scaled[:,-1]
p=pd.DataFrame(list(zip(xtest_scaled,ytest_scaled,predictions_scaled)),columns=['x','target_y','predictions'])
p=p.round(decimals=2)
p.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




