#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd

print("\nFile Type : .ipynb\n")

try:
    get_ipython().run_line_magic('load_ext', 'autoreload')      # %load_ext autoreload
    get_ipython().run_line_magic('autoreload', '2')             # %autoreload 2
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "all"
except:
    pass

#make necesarry imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation, cosine
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys, os
from contextlib import contextmanager
from pandas.core.reshape.pivot import pivot_table

from scipy.sparse.linalg import svds

import warnings
warnings.filterwarnings('ignore')


# In[34]:


def get_pivot(df):
    
    df = df.pivot_table(columns='book_id', index='user_id', values="rating").reset_index().rename_axis(None, axis=1)

    user_id_map = df['user_id']      # saving user index map for later reference
    df.drop(['user_id'],axis=1,inplace=True)
    
    return user_id_map, df

def get_similar_user_matrix(df, metric = 'cosine'):
    
    df.fillna(0,inplace=True)
    df = pd.DataFrame(1-pairwise_distances(df, metric=metric,force_all_finite=True))
    
    return df


# In[35]:


# variables
train_test_split_ratio = 0.7
user_count = 5                                # k nearest neighbors for users


# ### INPUT DATA

# In[20]:


interactions_df = pd.read_csv('../sample/interactions.csv', index_col=0).reset_index(drop=True)
interactions_df.drop(columns=['review_text_incomplete','read_at','started_at','review_id','is_read','date_updated'], inplace=True)

user_id_map_df = pd.read_csv('../sample/user_id_map.csv')

interactions_df = pd.merge(interactions_df,user_id_map_df,how="inner",on="user_id")

interactions_df.drop(columns=['user_id'], inplace=True)

interactions_df['user_id'] = interactions_df['user_id_csv']
interactions_df.drop(columns=['user_id_csv'], inplace=True)

books_df = pd.read_csv('../sample/top_1000_books.csv', index_col=0).reset_index(drop=True)
book_titles = books_df[['book_id','title']]

interactions_df = pd.merge(interactions_df, book_titles, on='book_id')

# interactions_df.head()


# In[1]:


# filtering out users who have given less than 50 ratings. 
x = interactions_df['user_id'].value_counts() > 50
y = x[x].index  #user_ids
interactions_df = interactions_df[interactions_df['user_id'].isin(y)]

interactions_df = interactions_df.reset_index()
interactions_df.drop(columns=['index'], inplace=True)
# interactions_df.head()

# filtering out books with less than 5 ratings. 
x = interactions_df['book_id'].value_counts() > 5
y = x[x].index  #user_ids
interactions_df = interactions_df[interactions_df['book_id'].isin(y)]

# sorting the data by date 
interactions_df['date_added'] = pd.to_datetime(interactions_df['date_added'])
interactions_df.sort_values('date_added', inplace = True)

# interactions_df.head()


# ## Choosing optimum k by train test

# #### TRAIN TEST SPLIT

# In[53]:


# train test split
split_value = int(train_test_split_ratio * len(interactions_df))
train_interaction = interactions_df[0:split_value]
test_interaction = interactions_df[split_value:]

# pivots
train_user_map, train_data = get_pivot(train_interaction)
test_user_map, test_data = get_pivot(test_interaction)

print('Percentage of overlapped users in train and test w.r.t. to train: ', 100*len(list(set(train_user_map).intersection(set(test_user_map))))/len(train_data))

train_data.to_csv('../Data/train_interaction_pivot.csv', index=False)
test_data.to_csv('../Data/test_interaction_pivot.csv', index=False)
train_user_map.to_csv('../Data/train_user_map.csv', index=False)
test_user_map.to_csv('../Data/test_user_map.csv', index=False)


# In[2]:


train_data = pd.read_csv('../Data/train_interaction_pivot.csv')
# train_data.fillna(0,inplace=True)
# train_data.sample(5)


# #### COSINE SIMILARITY 

# In[54]:


cosine_sim_train = get_similar_user_matrix(train_data, metric = 'cosine')
cosine_sim_train.to_csv('../Data/cosine_sim_train.csv', index=False)
cosine_sim_train.head()


# ####  Predictions for test data based on train model

# In[ ]:


test_data.fillna(0,inplace=True)

for user_count in range(50):
    result_df = pd.DataFrame(columns=['row','col','actual','predicted'])

    for col in range(test_data.shape[1]):
        temp = np.array(test_data.iloc[:,col]).nonzero()[0]

        for row in temp:
            book_id    = test_data.columns[col]

            try:
                user_id    = train_user_map[train_user_map == test_user_map[row]].index[0]
                user_index = list(train_data[train_data[book_id]>0][book_id].index)

                filtered_user = pd.DataFrame(cosine_sim_train.loc[user_id,user_index]).sort_values(user_id, ascending=False).head(user_count)

                predicted_rating = train_data[book_id][filtered_user.index].mean()   

                local_result = [row,col,test_data.iloc[row,col], predicted_rating]
                result_df.loc[len(result_df)] = local_result

            except:
                pass
    
    result_df['iteration'] = user_count
    result_df.to_csv('../output/user_cf_train_test_results/'+str(user_count)+'_user-user_cosine_sim_CF_test_result.csv', index = False)


# In[50]:


rmse_df = pd.DataFrame(columns=['k', 'rmse'])

for user_count in range(1,500):
    try:
        temp  = pd.read_csv('../output/user_cf_train_test_results/'+str(user_count)+'_user-user_cosine_sim_CF_test_result.csv')
        temp  = temp.dropna()
        rmse  = mean_squared_error(temp['actual'], temp['predicted'])
        local = [user_count,rmse]
        rmse_df.loc[len(rmse_df)] = local
    except:
        break
        
plt.plot(rmse_df['k'],rmse_df['rmse'])
plt.xlabel('k')
plt.ylabel('rmse')
plt.title('RMSE_vs_k')
plt.show()


# #### By elbow method, best k for the given model = 5

# ## Recommendation 

# In[51]:


user_count = 5


# In[36]:


user_map, data = get_pivot(interactions_df)

data.to_csv('../Data/full_interaction_pivot.csv', index=False)
user_map.to_csv('../Data/full_user_map.csv', index=False)

cosine_sim = get_similar_user_matrix(data, metric = 'cosine')
cosine_sim.to_csv('../Data/cosine_sim_full_data.csv', index=False)


# In[37]:


data.fillna(0,inplace=True)

result_df = pd.DataFrame(columns=['row','col','actual','predicted'])

def rating_by_user_cf(df,user):
    
    predicted_df = df.copy(deep=True)
    row = user
    
    for col in range(df.shape[1]):

        if(df.iloc[row,col]!=0):
            predicted_df[row,col] = None
            pass
        else:
            book_id    = df.columns[col]
            user_id    = row

            user_index = list(df[df[book_id]>0][book_id].index)

            filtered_user = pd.DataFrame(cosine_sim.loc[user_id,user_index]).sort_values(user_id, ascending=False).head(user_count)
                        
            predicted_rating = df[book_id][filtered_user.index].mean()   

            predicted_df.iloc[row,col] = predicted_rating
            
    return predicted_df, filtered_user


# In[38]:


# Recommending top 10 books based on ratings

def Recommend_book(df,user,number_of_books):
    '''
    Recommendation on basis of user-user similarity 
    '''
    
    predicted_ratings,similar_user = rating_by_user_cf(df,user)
        
    ls = list(pd.DataFrame(predicted_ratings.iloc[user,:]).sort_values(user, ascending=False).head(number_of_books).index)
    
    print('This user has rated',len(np.array(data.iloc[user,:]).nonzero()[0]), 'books.')
    
    return book_titles[book_titles['book_id'].isin(ls)]


# In[40]:


Recommend_book(data,user=0,number_of_books=10)


# ## MF using SVD

# ### Cross Validation

# In[96]:


# Taking 5 folds

index_df = pd.DataFrame(columns=['row','col','row_col'])
for col in range(data.shape[1]):
        temp = np.array(data.iloc[:,col]).nonzero()[0]
        
        for row in temp:
            
            local = [row,col,(row,col)]
            index_df.loc[len(index_df)] = local

index_df['Decile_rank'] = pd.qcut(index_df.index, 5, labels = False)


# In[3]:


# index_df


# In[90]:


ls = index_df['row_col'][index_df['Decile_rank'] == 1].reset_index(drop=True)

for cv_k in index_df['Decile_rank'].unique():
    
    ls = index_df['row_col'][index_df['Decile_rank'] == cv_k].reset_index(drop=True)
    
    cv_train = data.copy()
    for i in range(len(ls)):
        cv_train.iloc[ls[i]] = 0

    U, sigma, Vt = svds(cv_train, k = 50)

    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)   

    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = data.columns)
    # preds_df.head()

    y_actual = []
    y_pred = []
    for i in range(len(ls)):
        y_actual.append(data.iloc[ls[i]])
        y_pred.append(preds_df.iloc[ls[i]])

    print('Decile',cv_k, mean_squared_error(y_actual,y_pred))


# #### By cross validation, it's a stable model,because rmse for all five folds is almost same

# #### Choosing k (no of features)

# In[92]:


data.shape


# In[ ]:


ls = index_df['row_col'][index_df['Decile_rank'] == 1].reset_index(drop=True)

for cv_k in index_df['Decile_rank'].unique():
    
    ls = index_df['row_col'][index_df['Decile_rank'] == cv_k].reset_index(drop=True)
    
    data = data.copy()
    for i in range(len(ls)):
        data.iloc[ls[i]] = 0

    U, sigma, Vt = svds(data, k = 50)

    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)   

    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = data.columns)
    # preds_df.head()

    y_actual = []
    y_pred = []
    for i in range(len(ls)):
        y_actual.append(data.iloc[ls[i]])
        y_pred.append(preds_df.iloc[ls[i]])

    print('Decile',cv_k, mean_squared_error(y_actual,y_pred))


# In[98]:


ls = index_df['row_col'][index_df['Decile_rank'] == 1].reset_index(drop=True)
rmse_df = pd.DataFrame(columns=['k','rmse'])

for k in range(0,500,5):

    for cv_k in index_df['Decile_rank'].unique():
    
        ls = index_df['row_col'][index_df['Decile_rank'] == cv_k].reset_index(drop=True)

        data = data.copy()
        for i in range(len(ls)):
            data.iloc[ls[i]] = 0

        U, sigma, Vt = svds(data, k = 50)

        sigma = np.diag(sigma)

        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)   

        preds_df = pd.DataFrame(all_user_predicted_ratings, columns = data.columns)
        # preds_df.head()

        y_actual = []
        y_pred = []
        for i in range(len(ls)):
            y_actual.append(data.iloc[ls[i]])
            y_pred.append(preds_df.iloc[ls[i]])

        rmse_for_5_deciles.append(mean_squared_error(y_actual,y_pred))
    avg_rmse = sum(rmse_for_5_deciles)/len(rmse_for_5_deciles)
    print( 'k=',k, ':',avg_rmse)
        
    local = [k,avg_rmse]
    rmse_df.loc[len(rmse_df)] = local

    rmse_df.to_csv('../output/svd_cv_results/'+str(k)+'_cv5_svd_rmse.csv', index = False)


# In[103]:


# rmse_df

plt.plot(rmse_df['k'],rmse_df['rmse'])
plt.xlabel('k')
plt.ylabel('rmse')
plt.title('RMSE_vs_NFeatures')
plt.show()


# #### By elbow method, optimum number of features = 100

# ### Final Model

# In[40]:


# train_data.fillna(0,inplace=True)

# R = np.array(train_data)
# # R = train_data.as_matrix()
# user_ratings_mean = np.mean(R, axis = 1)
# user_ratings_std = np.std(R, axis=1)
# R_demeaned = R - user_ratings_mean.reshape(-1, 1)


# In[113]:


data = pd.read_csv('../Data/full_interaction_pivot.csv')
data


# In[47]:


def get_rating_by_svd(df,NFeatures=100):
    
    U, sigma, Vt = svds(data, k = 50)

    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = data.columns)
    
    return preds_df


# In[115]:


preds_df.head()


# In[45]:


# Recommending top 10 books based on ratings

def Recommend_book_svd(df,user,number_of_books):
    '''
    Recommendation on basis of svd 
    '''
    
    predicted_ratings = get_rating_by_svd(df)
        
    ls = list(pd.DataFrame(predicted_ratings.iloc[user,:]).sort_values(user, ascending=False).head(number_of_books).index.astype(int))
    print(ls)
    print('This user has rated',len(np.array(data.iloc[user,:]).nonzero()[0]), 'books.')
    
    return book_titles[book_titles['book_id'].isin(ls)]


# In[50]:


Recommend_book_svd(data,user=0,number_of_books=10)

