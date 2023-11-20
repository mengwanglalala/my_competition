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
import pickle
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
random.seed(2023)
np.random.seed(2023)
path = '../model/tmp_data/inputs/'

def display_importances(feature_importance_df_, name):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index

    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature",
                data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(f'lgbm_importances_{name}.png')


def gen_label(train):
    col = np.zeros((train.shape[0], 24))
    for i, label in enumerate(train['label'].values):
        col[i][label] = 1
    return col

label = pd.read_csv("../model/tmp_data/labels/training_label.csv")
lb_encoder = LabelEncoder()
label['label'] = lb_encoder.fit_transform(label['source'])

all_data=pd.read_parquet('../model/tmp_data/metric_feat.pqt').sort_values(by=['id']).reset_index(drop=True)
feats_tmp = pd.read_parquet('../model/tmp_data/trace_feat.pqt')
feats_tmp = feats_tmp.sort_values(by=['id']).reset_index(drop=True)

all_data = pd.concat([all_data,feats_tmp.drop(['id'],axis=1)],axis=1)

feats_tmp = pd.read_parquet('../model/tmp_data/log_feat.pqt')
feats_tmp = feats_tmp.sort_values(by=['id']).reset_index(drop=True)

all_data = pd.concat([all_data,feats_tmp.drop(['id'],axis=1)],axis=1)


all_data = all_data.merge(label[['id', 'label']].groupby(['id'], as_index=False)['label'].agg(list), how='left', on=['id']).set_index("id")


import pickle
with open('../other-file-or-dir/tmp_pickle/model2/sorted_feat.pickle', 'rb') as file:
    sorted_feat = pickle.load(file)

all_data=all_data[sorted_feat]

all_data = all_data.reindex(columns=sorted_feat)

for i in ["host_ip","service_name","span_id","service" ]:#, "service","endpoint_name", "service","trace_id","parent_id"
    lb = LabelEncoder()
    try:
        all_data[f"{i}_max_id"] = lb.fit_transform(all_data[f"{i}_max_id"])
    except:
        continue

data_add=pd.read_parquet('../model/tmp_data/encoder.pqt')
data=all_data.copy()

result = []
all_score = []
cores=62

for i in range(0, 24):
    class_ = i

    with open(f'../other-file-or-dir/tmp_pickle/model2/feat_{class_}.pickle', 'rb') as file:
        feat = pickle.load(file)

    all_data = data.copy()
    all_data = all_data[feat]

    y = gen_label(all_data[all_data['label'].notnull()])

    not_use = ['id', 'label', 'diff']
    feature_name = [i for i in all_data.columns if i not in not_use]

    print(f"Feature Length = {len(feature_name)}")
    num_classes = 24
    n_splits = 5

    kf = StratifiedKFold(n_splits=n_splits, random_state=3407, shuffle=True)
    #     kf = GroupKFold(n_splits=n_splits)
    scaler = StandardScaler()
    sts_feature = [i for i in feature_name if
                   i not in ['id', 'label', 'diff', "host_ip_max_id", "service_name_max_id", "trace_id_max_id",
                             "span_id_max_id", "parent_id_max_id", "service_max_id",
                             "trace_ip_max_id_service"]]  # , "service_max_id"
    all_data[sts_feature] = pd.DataFrame(
        scaler.fit_transform(all_data[sts_feature].fillna(0).replace([np.inf, -np.inf], 0)),
        index=all_data.index
    )

    scaler_X = all_data[feature_name].set_index(all_data.index)
    y = gen_label(all_data[all_data['label'].notnull()])
    train_scaler_X = scaler_X[all_data['label'].notnull()]
    test_scaler_X = scaler_X[all_data['label'].isnull()]
    train_scaler_X = pd.DataFrame(train_scaler_X, columns=feature_name)
    test_scaler_X = pd.DataFrame(test_scaler_X, columns=feature_name)

    train_scaler_X = pd.merge(train_scaler_X, data_add, on='id', how='left')
    test_scaler_X = pd.merge(test_scaler_X, data_add, on='id', how='left')

    feats = train_scaler_X.columns
    train_scaler_X = train_scaler_X.values
    test_scaler_X = test_scaler_X.values

    i = class_

    print("*" * 20, lb_encoder.classes_[i], "*" * 20)

    now_y = y[:, i]

    fold_score = []
    fold_num = 1
    ovr_preds = np.zeros(len(test_scaler_X))
    ovr_oof = np.zeros(len(train_scaler_X))
    feature_importance_df = pd.DataFrame()
    for train_index, valid_index in kf.split(train_scaler_X, now_y):
        X_train, X_valid = train_scaler_X[train_index], train_scaler_X[valid_index]
        #         X_train, X_valid = train_scaler_X.iloc[train_index], train_scaler_X.iloc[valid_index]
        y_train, y_valid = now_y[train_index], now_y[valid_index]

        clf = lgb.LGBMClassifier(objective='binary', n_estimators=500, num_leaves=63,
                                 subsample=0.8, colsample_bytree=0.25, subsample_freq=1,
                                 learning_rate=0.05, min_child_weight=1,
                                 random_state=2023, n_jobs=cores)
        clf.fit(X_train, y_train
                , eval_set=[(X_valid, y_valid)],
                eval_metric='auc',
                #                 early_stopping_rounds=100,
                verbose=500, )
        ovr_oof[valid_index] = clf.predict_proba(X_valid)[:, 1]
        # print(clf.predict_proba(test_scaler_X)[:,1] )
        ovr_preds += clf.predict_proba(test_scaler_X)[:, 1] / n_splits

        Score = roc_auc_score(y_valid, ovr_oof[valid_index])
        fold_score.append(Score)
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = fold_num + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        fold_num = fold_num + 1
    result.append(ovr_preds)
    # each_score = sScore(y, ovr_oof)
    print(f"{lb_encoder.classes_[i]} Mean Score = {np.mean(fold_score)}")
    all_score.append(fold_score)
#     display_importances(feature_importance_df,lb_encoder.classes_[i])

print(f"All Mean Score = {np.mean(all_score)}")

result = np.array(result).T
submit = pd.DataFrame(result, columns=lb_encoder.classes_)
submit.index = all_data[all_data['label'].isnull()].index
submit.reset_index(inplace=True)
submit = submit.melt(id_vars="id", value_vars=lb_encoder.classes_, value_name="score", var_name="source")
submit.to_csv(f"../model/tmp_data/model_2.csv", index=False)