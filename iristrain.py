
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


# In[3]:


irisdata = pd.read_excel('Iris.xls')
irisdata.head()


# In[5]:


#print(irisdata.describe())
cols = ['sepal length','sepal width','petal length','petal width','iris']
irisdata = irisdata[cols]
irisdata.head()


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


le = LabelEncoder()
le.fit(irisdata['iris'].astype(str))
irisdata['iris']=le.transform(irisdata['iris'].astype(str))
irisdata.head()


# In[7]:


import matplotlib.pyplot as plt

jc = {'0':100,'1':170,'2':220}
g = irisdata['iris']
map(g,jc)
plt.scatter(irisdata['sepal length'],irisdata['sepal width'],c = g)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Plot')
plt.show()


# In[8]:


plt.subplot(221)
plt.hist(irisdata['iris'])
plt.subplot(222)
plt.hist(irisdata['petal length'])
plt.subplot(223)
plt.hist(irisdata['petal width'])
plt.subplot(224)
plt.hist(irisdata['sepal width'])
plt.show()


# In[11]:


sns.pairplot(irisdata,hue="iris",vars = cols[0:4])


# In[13]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix


# In[61]:


x,x_t,y,y_t = train_test_split(irisdata[cols[0:4]],irisdata['iris'],test_size=0.2)


# In[62]:


#LinearSVC() it uses one versus the rest..
tr = svm.LinearSVC()
tr.fit(x,y)
h = tr.predict(x_t)

cfm = confusion_matrix(y_t,h)
print(tr.score(x_t,y_t))
print(cfm)


# In[63]:


#RandomForestClassifier()
tr = RandomForestClassifier(n_estimators = 100)
tr.fit(x,y)
h = tr.predict(x_t)

cfm = confusion_matrix(y_t,h)
print(tr.score(x_t,y_t))
print(cfm)


# In[65]:


#DecisionTreeClassifier()
from sklearn.tree import DecisionTreeClassifier

tr = DecisionTreeClassifier()
tr.fit(x,y)
h = tr.predict(x_t)

cfm = confusion_matrix(y_t,h)
print(tr.score(x_t,y_t))
print(cfm)


# In[69]:


#KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

tr = KNeighborsClassifier()
tr.fit(x,y)
h = tr.predict(x_t)

cfm = confusion_matrix(y_t,h)
print(tr.score(x_t,y_t))
print(cfm)

