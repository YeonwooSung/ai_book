
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
import os
import glob


# In[3]:


df = pd.read_hdf("D:\Mtech\Sem4\ASR\Project\Feature\\test_timit_mfcc_delta_delta.hdf")
features = np.array(df["features"].tolist())
label = np.array(df["labels"].tolist())
df1 = pd.DataFrame(features)
df2 = pd.DataFrame(label)
df = pd.merge(df1, df2, right_index=True, left_index=True)
df.rename(columns={'0_y': 'Label'}, inplace=True)
print(df.head())


# In[4]:


perCt =0
totalCt =0

def calPER(x,lbl):
    global perCt
    global totalCt
    totalCt += 1
    if x[lbl] != x['Label']:
        #print(x[lbl],x['Label'])
        if(len(x[lbl]) == 0 and len(x['Label'])!= 0 ) or (len(x[lbl]) != 0 and len(x['Label'])== 0 ) or (len(x[lbl]) != 0 and len(x['Label'])!= 0 ):
            perCt += 1

        
test_df = df.iloc[:,:-1]
for i in range(1,8):
    filename = ''.join(['D:\Mtech\Sem4\ASR\Project\Model\\',str(2**i),'M_GMM\\','*.pkl'])
    eval_df = pd.DataFrame()
    for file in glob.glob(filename):
        #print(file)
        with open(file, "rb") as f:
            gmm = pickle.load(f)
            score = gmm.score_samples(test_df)
            label = os.path.basename(file).split('_')[0]
            eval_df[label] = score
    eval_label = ''.join(['eval_score_',str(2**i),'M_GMM'])
    df[eval_label] = eval_df.idxmax(axis=1)
    df.apply(calPER,axis=1,args=(eval_label,))
    print('PER for',str(2**i),'M_GMM',perCt/totalCt)
    perCt =0
    totalCt =0

