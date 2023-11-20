import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm

from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn import base
import random
import joblib

random.seed(2023)
np.random.seed(2023)
path = '../data/inputs'


class KFoldTargetEncoderTrain(base.BaseEstimator,
                              base.TransformerMixin):
    def __init__(self, colnames, targetName,
                 n_fold=5, verbosity=True,
                 discardOriginal_col=False):
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert (type(self.targetName) == str)
        assert (type(self.colnames) == str)
        assert (self.colnames in X.columns)
        assert (self.targetName in X.columns)
        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=2019)
        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan
        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(
                X_tr.groupby(self.colnames)[self.targetName].mean())
            X[col_mean_name].fillna(mean_of_target, inplace=True)
        if self.verbosity:
            encoded_feature = X[col_mean_name].values
        #             print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,self.targetName,np.corrcoef(X[self.targetName].values,encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
        return X


class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, train, colNames, encodedName):
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mean = self.train[[self.colNames,
                           self.encodedName]].groupby(
            self.colNames).mean().reset_index()
        #         print(X[self.colNames])
        #         X = X.merge(mean,on=self.colNames,how="left")
        #         print(mean)
        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]
        X[self.encodedName] = X[self.colNames]
        X = X.replace({self.encodedName: dd})
        return X


def sScore(y_true, y_pred):
    score = []
    for i in range(num_classes):
        score.append(roc_auc_score(y_true[:, i], y_pred[:, i]))

    return score

def gen_label(train):
    col = np.zeros((train.shape[0], 24))
    for i, label in enumerate(train['label'].values):
        col[i][label] = 1
    return col


label = pd.read_csv("../model/tmp_data/labels/training_label.csv")
lb_encoder = LabelEncoder()
label['label'] = lb_encoder.fit_transform(label['source'])


all_data=pd.read_parquet('../model/tmp_data/trace_feat.pqt').sort_values(by=['id']).reset_index(drop=True)
feats_tmp = pd.read_parquet('../model/tmp_data/log_feat.pqt')
feats_tmp = feats_tmp.sort_values(by=['id']).reset_index(drop=True)

all_data = pd.concat([all_data,feats_tmp.drop(['id'],axis=1)],axis=1)

all_data = all_data.merge(label[['id', 'label']].groupby(['id'], as_index=False)['label'].agg(list), how='left', on=['id']).set_index("id")



y = gen_label(all_data[all_data['label'].notnull()])
for i in ["host_ip","service_name","span_id","service" ]:#, "service","endpoint_name", "service","trace_id","parent_id"
    lb = LabelEncoder()
    try:
        all_data[f"{i}_max_id"] = lb.fit_transform(all_data[f"{i}_max_id"])
    except:
        continue
#
# lb = LabelEncoder()
# all_data[f"trace_ip_max_id_service"] = lb.fit_transform(all_data[f"trace_ip_max_id_service"])

#
not_use = ['id', 'label', 'diff']
feature_name = [i for i in all_data.columns if i not in not_use]



# for i in feature_name:
#     print(all_data[i].dtype,i)
print(f"Feature Length = {len(feature_name)}")
# print(f"Feature = {feature_name}")
num_classes = 24
n_splits = 5



kf = StratifiedKFold(n_splits=n_splits, random_state=3407, shuffle=True)
scaler = StandardScaler()
# scaler_X = scaler.fit_transform(X.fillna(0).replace([np.inf, -np.inf], 0))
sts_feature = [i for i in feature_name if
               i not in ['id', 'label', 'diff', "host_ip_max_id", "service_name_max_id", "trace_id_max_id",
                         "span_id_max_id", "parent_id_max_id", "service_max_id",
                         "trace_ip_max_id_service"]]  # , "service_max_id"
all_data[sts_feature] = pd.DataFrame(
    scaler.fit_transform(all_data[sts_feature].fillna(0).replace([np.inf, -np.inf], 0)),
    index=all_data.index
)  # .set_index(all_data.index)
# print(all_data)
# print(all_data["host_ip_max_id"])
scaler_X = all_data[feature_name].set_index(all_data.index)
# for i in range(len(lb_encoder.classes_)):
y = gen_label(all_data[all_data['label'].notnull()])
train_scaler_X = scaler_X[all_data['label'].notnull()]
test_scaler_X = scaler_X[all_data['label'].isnull()]
train_scaler_X = pd.DataFrame(train_scaler_X, columns=feature_name)
test_scaler_X = pd.DataFrame(test_scaler_X, columns=feature_name)
for i in range(24):
    #     temp_train_df = pd.DataFrame(train_scaler_X, columns=feature_name)
    #     temp_train_df['label'] = y[:i]
    temp_y = y[:, i]
    train_scaler_X["label"] = temp_y
    targetc = KFoldTargetEncoderTrain("host_ip_max_id", "label", n_fold=5)
    new_train = targetc.fit_transform(train_scaler_X)
    test_targetc = KFoldTargetEncoderTest(new_train, "host_ip_max_id", "host_ip_max_id_Kfold_Target_Enc")
    new_test = test_targetc.fit_transform(test_scaler_X)
    train_scaler_X[f"host_ip_max_id_Kfold_Target_Enc_{i}"] = new_train["host_ip_max_id_Kfold_Target_Enc"]
    test_scaler_X[f"host_ip_max_id_Kfold_Target_Enc_{i}"] = new_test["host_ip_max_id_Kfold_Target_Enc"]
    del train_scaler_X["host_ip_max_id_Kfold_Target_Enc"]
    del test_scaler_X["host_ip_max_id_Kfold_Target_Enc"]

    targetc = KFoldTargetEncoderTrain("service_name_max_id", "label", n_fold=5)
    new_train = targetc.fit_transform(train_scaler_X)
    test_targetc = KFoldTargetEncoderTest(new_train, "service_name_max_id", "service_name_max_id_Kfold_Target_Enc")
    new_test = test_targetc.fit_transform(test_scaler_X)
    train_scaler_X[f"service_name_max_id_Kfold_Target_Enc_{i}"] = new_train["service_name_max_id_Kfold_Target_Enc"]
    test_scaler_X[f"service_name_max_id_Kfold_Target_Enc_{i}"] = new_test["service_name_max_id_Kfold_Target_Enc"]
    del train_scaler_X["service_name_max_id_Kfold_Target_Enc"]
    del test_scaler_X["service_name_max_id_Kfold_Target_Enc"]

    targetc = KFoldTargetEncoderTrain("span_id_max_id", "label", n_fold=5)
    new_train = targetc.fit_transform(train_scaler_X)
    test_targetc = KFoldTargetEncoderTest(new_train, "span_id_max_id", "span_id_max_id_Kfold_Target_Enc")
    new_test = test_targetc.fit_transform(test_scaler_X)
    train_scaler_X[f"span_id_max_id_Kfold_Target_Enc_{i}"] = new_train["span_id_max_id_Kfold_Target_Enc"]
    test_scaler_X[f"span_id_max_id_Kfold_Target_Enc_{i}"] = new_test["span_id_max_id_Kfold_Target_Enc"]
    del train_scaler_X["span_id_max_id_Kfold_Target_Enc"]
    del test_scaler_X["span_id_max_id_Kfold_Target_Enc"]

    targetc = KFoldTargetEncoderTrain("service_max_id", "label", n_fold=5)
    new_train = targetc.fit_transform(train_scaler_X)
    test_targetc = KFoldTargetEncoderTest(new_train, "service_max_id", "service_max_id_Kfold_Target_Enc")
    new_test = test_targetc.fit_transform(test_scaler_X)
    train_scaler_X[f"service_max_id_Kfold_Target_Enc_{i}"] = new_train["service_max_id_Kfold_Target_Enc"]
    test_scaler_X[f"service_max_id_Kfold_Target_Enc_{i}"] = new_test["service_max_id_Kfold_Target_Enc"]
    del train_scaler_X["service_max_id_Kfold_Target_Enc"]
    del test_scaler_X["service_max_id_Kfold_Target_Enc"]

del train_scaler_X["label"]

encoder=pd.concat([train_scaler_X.iloc[:,-96:],test_scaler_X.iloc[:,-96:]],axis=0)

encoder.to_parquet('../model/tmp_data/encoder.pqt')