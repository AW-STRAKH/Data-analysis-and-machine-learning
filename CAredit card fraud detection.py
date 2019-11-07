#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn




print('python:{}'.format(sys.version))
print('pandas:{}'.format(pandas.__version__))
print('numpy:{}'.format(numpy.__version__))
print('SKLEARN:{}'.format(sklearn.__version__))



# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv(r'C:\Users\Aw-Strakh\Desktop\creditcard.csv')


# In[4]:


print(data.columns)  
data = data.sample(frac = 0.3, random_state = 1)
print (data.shape)


# In[5]:


data = pd.read_csv(r'C:\Users\Aw-Strakh\Desktop\creditcard.csv')


# In[6]:


print(data.shape)


# In[ ]:





# In[7]:


data.hist(figsize=(20,20))
plt.show()


# In[8]:


fraud=data[data['Class']==1]
valid=data[data['Class']==0]
outlier_frac=len(fraud)/float(len(valid))
print(outlier_frac)
print(len(fraud))
print(len(valid))


# In[9]:


#correlation-to find between vaiables of dadtaset
cormat=data.corr()


# In[10]:


fig=plt.figure(figsize=(12,9))
sns.heatmap(cormat,vmax=.8,square=True)
plt.show()


# In[11]:


#removig class coz its unsupervised
columns =data.columns.tolist()
columns=[c for c in columns if c not in ["Class"]]
target = "Class"
x=data[columns]
y=data[target]
print(x.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


from sklearn.metrics import  classification_report, accuracy_score
from sklearn.ensemble import IsolationForest     #split value of a feature
from sklearn.neighbors import LocalOutlierFactor #unsupervised local deviation of density from neighbours as k nearest neighbour
#average of path lengths=normality

state = 1
classifier={"IF": IsolationForest(max_samples=len(x),contamination=outlier_frac,random_state=state),
            "LOF":LocalOutlierFactor(n_neighbors=20,contamination = outlier_frac )
           }
            
n_out=len(fraud)
for i, (clf_nm,clf) in enumerate(classifier.items()):
    if clf_nm == "LOF":
        y_pred = clf.fit_predict(x)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(x)
        scores_pred = clf.decision_function(x)
        y_pred = clf.predict(x)
        
        
# making y_pred same as y as it will give in the form of -1 and 1 while required is in the form of 0 and 1
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1

    n_errors=(y_pred!=y).sum()
    print('{}.{}'.format(clf_nm,n_errors))
    print(accuracy_score(y,y_pred))
    print(classification_report(y,y_pred))
        
        


# In[ ]:





# In[ ]:





# In[ ]:




