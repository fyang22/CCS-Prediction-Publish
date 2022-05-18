#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libaries
import numpy as np
import pandas as pd # Version 0.24.2 to use rdkit PandasTools modules
pd.show_versions()


# In[2]:


# load data
df_all = pd.read_csv(r'C:\Users\fyang\CCS-Prediction-paper\Data\alldata.csv')


# In[3]:


from rdkit import Chem
from rdkit.Chem import PandasTools

# get fingerprints
def get_FPs(df):
    
    # from SMILES to mol
    PandasTools.AddMoleculeColumnToFrame(df,
                                         'SMILES','Molecules',
                                         includeFingerprints = True)
    df.dropna(subset = ['Molecules'],inplace = True)
    
    # from mol to FPs
    df['FPs'] = [Chem.RDKFingerprint(mol,2, fpSize = 1024) for mol in df.Molecules]
    
    return df


# In[4]:


df_all = get_FPs(df_all)


# In[5]:


# Save FPs with each bit as a column for futher use
def save_FPs(df): 
    
    FPs_name = [f'Bit_{i}' for i in range(1024)]
    fps = [list(fp) for fp in df.FPs]
    df_FPs = pd.DataFrame(fps, columns = FPs_name)   
    
    return df_FPs


# In[6]:


df_allFPs = save_FPs(df_all)
df_allsave = pd.concat([df_all, df_allFPs], axis = 1, ignore_index = False,sort=False)


# In[7]:


# select data for classifier modeling
df_class = df_all.dropna(subset = ['Super.Class'])

# clean up data
df_class.drop_duplicates(subset = ['CID'], inplace = True)

# check defined super class
df_class['Super.Class'].value_counts()


# In[8]:


# split df by super class
gb_class = df_class.groupby('Super.Class')
listOfClass = [gb_class.get_group(x) for x in gb_class.groups]


# In[9]:


#find the top5 biggest class
listOfClass.sort(key=(lambda x: x.shape[0]),reverse=True)


# In[10]:


# calculate FPs similarity and combine classes
from rdkit import DataStructs

def combine_class(list_Of_df,similarity_score):
    for i in range(5): #choose the top 5 biggest class as references
        print(i)
        df_ref = list_Of_df[i] #get df from list
        FPs_ref = df_ref['FPs'] #get ref FPs
        label = df_ref.iloc[0]['Super.Class'] #store class label
        
        for j in range(5,len(list_Of_df)): # classes to be combined
            df = list_Of_df[j]
            FPs = df['FPs']
            for fp_ref in FPs_ref:
                for fp in FPs:
                    score = DataStructs.FingerprintSimilarity(fp_ref, fp)
                    if score >= similarity_score:
                        df.loc[df.FPs == fp, 'Super.Class'] = label # assigne to new class
    return list_Of_df


# In[ ]:


listOfClass = combine_class(listOfClass, 0.2)


# In[ ]:


df_class = pd.concat(listOfClass)
df_class.reset_index(drop = True, inplace = True)


# In[ ]:


df_class['Super.Class'].value_counts()


# In[ ]:


# PCA 
from sklearn.preprocessing import StandardScaler 

df_classFPs = save_FPs(df_class)
x = df_classFPs # features
y = df_class.loc[:,['Super.Class']].values # target

# Standardizing the features
x = StandardScaler().fit_transform(x)


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
decomp = pca.fit_transform(x)
df_decomp = pd.DataFrame(data = pc, columns = ['PC_1', 'PC_2'])


# In[ ]:


labels = (df_class['Super.Class'].value_counts()).index


# In[ ]:


# plot PCA
import matplotlib.pyplot as plt

colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd']
fig = plt.figure(figsize= (8,8))
for name, color in zip(labels, colors):
    indices = df_class['Super.Class'] == name
    plt.scatter(df_decomp.loc[indices, 'PC_1'],
               df_decomp.loc[indices,'PC_2'], c = color, s =50)
    plt.legend(labels, prop = {'size':8}, )
plt.title('2 component PCA, FPs with class')
plt.xlabel('Principal Component 1',fontsize = 12)
plt.ylabel('Principal Component 2',fontsize = 12)
plt.show()
#fig.savefig('PCA_FPs.png',bbox_inches = 'tight', dpi = 100 )


# In[ ]:




