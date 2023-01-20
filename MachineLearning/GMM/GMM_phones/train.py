
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
import os


# In[2]:


df = pd.read_hdf("D:\Mtech\Sem4\ASR\Project\Feature\\timit_mfcc_delta_delta.hdf")
features = np.array(df["features"].tolist())
label = np.array(df["labels"].tolist())
df1 = pd.DataFrame(features)
df2 = pd.DataFrame(label)
df = pd.merge(df1, df2, right_index=True, left_index=True)
df.rename(columns={'0_y': 'Label'}, inplace=True)
print(df.head())


# In[3]:


labels=df.Label.unique()
print(labels)
print(len(labels))
labels= pd.Series(labels)
os.mkdir('D:\Mtech\Sem4\ASR\Project\Model\\2M_GMM')
os.mkdir('D:\Mtech\Sem4\ASR\Project\Model\\4M_GMM')
os.mkdir('D:\Mtech\Sem4\ASR\Project\Model\\8M_GMM')
os.mkdir('D:\Mtech\Sem4\ASR\Project\Model\\16M_GMM')
os.mkdir('D:\Mtech\Sem4\ASR\Project\Model\\32M_GMM')
os.mkdir('D:\Mtech\Sem4\ASR\Project\Model\\64M_GMM')
os.mkdir('D:\Mtech\Sem4\ASR\Project\Model\\128M_GMM')


# In[5]:


def train_gmm(x):
    print(type(x),x)
    global df
    for i in range(1,8):
        array = df.loc[df['Label']==x].iloc[:,:-1]
        gmm = GaussianMixture(n_components=2**i,covariance_type='diag').fit(array)
        file = ''.join(['D:\Mtech\Sem4\ASR\Project\Model\\',str(2**i),'M_GMM\\',x,'_', str(2**i) ,'.pkl'])
        print(file)
        with open(file, 'wb') as f:
            pickle.dump(gmm, f)

labels.apply(train_gmm)

