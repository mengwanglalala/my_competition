import pandas as pd
import numpy as np
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold
import lightgbm as lgb
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
#%matplotlib inline
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error, log_loss
pd.options.display.precision = 15
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
import time
import datetime
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from sklearn import metrics
from sklearn import linear_model
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import json
import copy
#%matplotlib inline

#%env JOBLIB_TEMP_FOLDER=/tmp


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
    
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(y_true, y_pred):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', fast_auc(y_true, y_pred), True


def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    """
    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling
    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric
    """
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()
    



train_base = pd.read_csv('/data/train_base.csv')
test_base = pd.read_csv('/data/testa_base.csv')

train_base = train_base.dropna(subset=['age'])
train_base


train_fea = pd.read_csv('fea1.csv')
print(len(train_fea))
train_fea.head()

trian_data = pd.merge(train_base, train_fea, on=['cust_wid'],how = 'left')
print(len(trian_data))
trian_data.head()

train  = trian_data
train['label'][train.label>=1]=1
train = train.fillna(0)
print(train.shape)
train.head()

test_fea =pd.read_csv('test_fea1.csv')
print(test_fea.shape)
test = pd.merge(test_base, test_fea, on=['cust_wid'],how = 'left')
test = test.fillna(0)
print(test.shape)
test.head()


cat_cols = ['gdr_cd','cty_cd']
question_cols = []
continue_cols = []
for col in train.columns:
    if col not in ['cust_wid','label']+cat_cols:
        continue_cols.append(col)

for col in cat_cols:
    if col in train.columns:
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))  
train.head()

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
n_fold = 5
#folds = TimeSeriesSplit(n_splits=n_fold)
folds = KFold(n_splits=5)

#continue_cols+ ['CUST_UID','LABEL']+cat_cols+question_cols+large_cale_cols
X = train.sort_values('cust_wid').drop(['label', 'cust_wid'], axis=1)
y = train.sort_values('cust_wid')['label']
X_test = test.drop(['cust_wid'], axis=1)

def clean_inf_nan(df):
    return df.replace([np.inf, -np.inf], np.nan)   

# Cleaning infinite values to NaN
X = clean_inf_nan(X)
X_test = clean_inf_nan(X_test )


print(len(X.columns))
print(X.shape)

averag = 'usual'#'rank'#'usual'#
model_name = 'lgb'
params = {'num_leaves': 63,#256,#
          'min_child_samples': 79,
          'objective': 'binary',
          'max_depth': -1,#7,#15
          'learning_rate': 0.03,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          "subsample": 0.75,#0.9
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 0.9#,
          #'categorical_feature': cat_cols
         }

# 预测函数
import pickle
def pkl_save(filename,file):
    output = open(filename, 'wb')
    pickle.dump(file, output)
    output.close()

def pkl_load(filename):
    pkl_file = open(filename, 'rb')
    file = pickle.load(pkl_file) 
    pkl_file.close()
    return file


def run_test(X_test,X):
    result_dict = {}
    columns_test = X.columns 
    print(X.columns)
    n_splits = 5
    X_test = X_test[columns_test]

    models = [pkl_load("models/model-{}.lgb".format(idx)) for idx in range(n_splits)]
    prediction = np.zeros((len(X_test), 1))
    for idx,model in enumerate(models):
        y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]
        prediction += y_pred.reshape(-1, 1)
    
    prediction /= n_splits
    result_dict['prediction'] = prediction
    return result_dict

result_dict_lgb = run_test(X_test,X)

test["label"] = result_dict_lgb['prediction']
columns = ["cust_wid","label"]
results = test[columns]
results.head()

print(len(results[results['label'] >= 0.2])/len(results))
# results['label'][results['label'] > 0.45] = 1
# results['label'][results['label'] <= 0.45] = 0
results['label'][results['label'] >=0.2] = 1
results['label'] = results['label'].astype(int)
results = results.sort_values('cust_wid')
results = results.reset_index(drop = True)
results.head()

results.to_csv('./output.csv',index = 0)
print('predic finished')