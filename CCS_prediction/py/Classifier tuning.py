#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# load data
df_class = pd.read_csv(r'C:\Users\fyang\CCS-Prediction-Publish\Data\classifier_model_data.csv')


# In[3]:


ind_FP0 = df_class.columns.get_loc("Bit_0")
ind_FP0


# In[4]:


# Random forest classifier modeling
from sklearn.model_selection import train_test_split

X_SC = df_class.iloc[:,ind_FP0:]
y_SC = df_class['Super.Class'].values

X_SC_train, X_SC_test, y_SC_train, y_SC_test = train_test_split(X_SC, y_SC, 
                                                                train_size = 0.8, 
                                                                test_size=0.2, 
                                                                random_state=101,
                                                                stratify=y_SC)

print('X_SC_train: {}'.format(np.shape(X_SC_train)))
print('y_SC_train: {}'.format(np.shape(y_SC_train)))
print('X_SC_test: {}'.format(np.shape(X_SC_test)))
print('y_SC_test: {}'.format(np.shape(y_SC_test)))


# In[5]:


# check training data
unique, counts = np.unique(y_SC_train, return_counts=True)
print (np.asarray((unique, counts)).T)


# In[6]:


# check test data
unique_test, counts_test = np.unique(y_SC_test, return_counts=True)
print (np.asarray((unique_test, counts_test)).T)


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# In[8]:


# set parameters for tuning

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 200, num = 5)]
# Quality of a split
criterion = ['gini']
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [None]
# Minimum number of samples required to split a node
# min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 5, num =3)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(start = 5, stop = 15, num =5)]
# Use out-of-bag samples to estimate the generalization score. 
oob_score = [True]


# In[9]:


# Creat the param grid
param_grid = {'n_estimators': n_estimators,
              #'criterion': criterion,
              #'max_features': max_features,
              #'max_depth': max_depth,
              #'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
             # 'oob_score': oob_score
               }
# Define model type
rfc_Model = RandomForestClassifier(oob_score = [True])


# In[10]:


rfc_Grid = GridSearchCV(estimator = rfc_Model, 
                       param_grid = param_grid, 
                       cv = 5, 
                       verbose=2, 
                       n_jobs = 3,
                       return_train_score = True
                       )


rfc_Grid.fit(X_SC_train, y_SC_train)


# In[11]:


# Optimized method 
print(rfc_Grid.best_params_)
print (f'Train Accuracy - : {rfc_Grid.score(X_SC_train,y_SC_train):.3f}')
print (f'Test Accuracy - : {rfc_Grid.score(X_SC_test,y_SC_test):.3f}')


# In[12]:


labels = ['Organic acids and derivatives', 
                        'Lipids and lipid-like molecules',
                        'Benzenoids', 
                        'Organic oxygen compounds',
                        'Organoheterocyclic compounds']


# In[13]:


# F1 Scores
from sklearn.metrics import f1_score, make_scorer
f1 = make_scorer(f1_score, average = None, labels = labels)

pred=rfc_Grid.predict(X_SC_test)

f1_score = f1_score(y_SC_test, pred, average= None,
             labels = labels)
f1_score


# In[14]:


import matplotlib.pyplot as plt
# Plot results

def plot_search_results(grid):

    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params=grid.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='-', marker='o', label='test')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()


# In[15]:


plot_search_results(rfc_Grid)


# In[16]:


#Evaluation confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
matrix = confusion_matrix(y_SC_test, pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns
# Build the plot
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':10},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
labels # SuperClass name

tick_marks = np.arange(len(labels))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, labels, rotation=25)
plt.yticks(tick_marks2, labels, rotation=0)
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.title('Confusion Matrix for Random Forest Classifier Model')
plt.savefig('Confusion Matrix.png',bbox_inches='tight')
plt.show()


# In[18]:


# get index of test datasets
l_SCtest_index = list(X_SC_test.index.values)
# get test dataset smiles
l_SCtest_smiles = df_class['SMILES'].loc[l_SCtest_index]
l_SCtest_names = df_class['name'].loc[l_SCtest_index]


# In[19]:


# Check wrongly predicted chemicals
dict = {'name':l_SCtest_names, 'SMILES':l_SCtest_smiles, 'SuperClass':y_SC_test, 'Prediction':pred}
df_evalu = pd.DataFrame(dict)


# In[20]:


# Compare predicted to defined class
df_evalu['comparison_column'] = np.where(df_evalu["SuperClass"] == df_evalu["Prediction"], 
                                           True, False)


# In[21]:


df_evalu.drop(df_evalu[df_evalu['comparison_column'] == True].index, 
              inplace = True)


# In[22]:


# randomly choose 3 confusion predicted chemicals from each class
wrongPred = df_evalu.groupby('SuperClass').apply(pd.DataFrame.sample, n = 3)


# In[23]:


# get molecules
from rdkit import Chem
df_class['Molecule'] = df_class['SMILES'].apply(Chem.MolFromSmiles)


# In[24]:


# get index of random choose chemicals
l_wrongPred_index = [i[1] for i in list(wrongPred.index.values)]
for i in l_wrongPred_index:
    print(df_class['name'][i])
    display(df_class['Molecule'][i])


# In[25]:


# Feature importance
opt_model = rfc_Grid.best_estimator_ # best optimized model
model_FI = opt_model.feature_importances_


# In[26]:


# investigate the top 10 importance features
ind = np.argpartition(model_FI,-10)[-10:] 
print(ind)
print(model_FI[ind])


# In[27]:


from rdkit.Chem import Draw

# get bits which have top10 highest importance features
FPs_columns = [f'Bit_{i}' for i in ind]

# collect chemicals with the presence of selected features
listOf_df_FI = [df_class.loc[df_class[column] == 1] for column in FPs_columns]


# In[28]:


# get feature examples
def get_svgs(df,FI_index):
    
    rdkbi = {}
    
    for mol in df['Molecule']:
        rdkfp = Chem.RDKFingerprint(mol, 2, fpSize = 1024, bitInfo=rdkbi)
        svg = Draw.DrawRDKitBit(mol, FI_index, rdkbi, useSVG=True)
        return(svg)


# In[29]:


# display features
from IPython.core.display import display
df_FI = listOf_df_FI[0]
df_sampleFI = df_FI.sample(n = 1)    
svg = get_svgs(df_sampleFI, ind[0])
print(ind[0])
display(svg)


# In[30]:


# save the model
import joblib
filename = 'classifier_prediction.joblib'
joblib.dump(rfc_Grid, filename)


# In[ ]:




