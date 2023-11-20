import os
import warnings

warnings.filterwarnings('ignore')
import random
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm.auto import tqdm
from xgboost import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


random.seed(2023)
np.random.seed(2023)
def sScore(y_true, y_pred):
    score = []
    for i in range(num_classes):
        score.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        
    return score


import json


def extract_metric_name(json_str):
    # 将JSON格式的字符串解析为Python字典
    data = json.loads(json_str)

    # 提取"metric_name"字段的值
    metric_name = data.get("metric_name")

    return metric_name


def extract_kubernetes_io_hostname(json_str):
    # 将JSON格式的字符串解析为Python字典
    data = json.loads(json_str)

    # 提取"metric_name"字段的值
    metric_name = data.get("kubernetes_io_hostname")

    return metric_name


def extract_container(json_str):
    # 将JSON格式的字符串解析为Python字典
    data = json.loads(json_str)

    # 提取"metric_name"字段的值
    metric_name = data.get("container")

    return metric_name
def extract_service_name(json_str):
    data = json.loads(json_str)
    metric_name = data.get("service_name")

    return metric_name

def extract_instance_name(json_str):
    data = json.loads(json_str)
    metric_name = data.get("instance")

    return metric_name


def extract_pod_name(json_str):
    data = json.loads(json_str)
    metric_name = data.get("pod")

    return metric_name
def change_pod(x):
    temp=str(x).split('-')
    if temp[-1] in ['master','0','1','2'] or len(temp)==1:
        return x
    else:
        return temp[0]
    
def processing_feature(file):
    log, trace, metric, metric_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
#     if os.path.exists(f"../data/tmp_data/inputs/log/{file}_log.parquet"):
#         log = pd.read_parquet(f"../data/tmp_data/inputs/log/{file}_log.parquet").sort_values(by=['timestamp']).reset_index(drop=True)

#     if os.path.exists(f"../data/tmp_data/inputs/trace/{file}_trace.parquet"):
#         trace = pd.read_parquet(f"../data/tmp_data/inputs/trace/{file}_trace.parquet").sort_values(by=['timestamp']).reset_index(drop=True)

    if os.path.exists(f"../model/tmp_data/inputs/metric/{file}_metric.csv"):
        metric = pd.read_csv(f"../model/tmp_data/inputs/metric/{file}_metric.csv").sort_values(by=['timestamp']).reset_index(drop=True)
        
        
   
        
        
    feats = {"id": file}

    if len(metric) > 0:
        feats['metric_length'] = len(metric)
        feats['metric_value_timestamp_value_mean_std'] = metric.groupby(['timestamp'])['value'].mean().std()
        metric['metric_name']=metric['tags'].apply(extract_metric_name)
        metric['hostname']=metric['tags'].apply(extract_kubernetes_io_hostname)
        metric['container']=metric['tags'].apply(extract_container)
        metric['service_name'] = metric['tags'].apply(extract_service_name)
        metric['instance'] = metric['tags'].apply(extract_instance_name)
        metric['pod'] = metric['tags'].apply(extract_pod_name)
        metric['pod'] = metric['pod'].apply(change_pod)
        
        
        
        for metric_name,df in list(metric.groupby('metric_name')):
            if len(df)>0:
                feats[f'metric_name_cnt_{metric_name}'] = len(df)
#                 df_temp=df.groupby(['timestamp'])['value']#.mean()
                
                for stats_func in ['mean','ptp']:#,'std', 'skew', 'kurt',
                    feats[f"metric_timestamp_value_{stats_func}_{metric_name}"] = df["value"].agg(stats_func)
                    
#                 df['diff']=df['timestamp'].diff()
#                 for stats_func in ['mean','std', 'skew', 'kurt','ptp']:
#                     feats[f"metric_diff_{stats_func}_{metric_name}"] =df['diff'].apply(stats_func)



        
        for (metric_name,container,pod),df in list(metric.groupby(['metric_name','container','pod'])):
            if len(df)>0:
                feats[f'metric_pod_name_cnt_{metric_name}_{container}_{pod}'] = len(df)
                for stats_func in ['mean','ptp']:
                    feats[f"metric_pod_name_timestamp_value_{stats_func}_{metric_name}_{container}_{pod}"] = df['value'].agg(stats_func)
        
    
#         for (container,pod),df in list(metric.groupby(['container','pod'])):
#             if len(df)>0:
#                 feats[f'metric_pod_container_cnt_{container}'] = len(df)
#                 df_temp=df.groupby(['metric_name'])['value'].mean()#,'timestamp'
                
#                 for stats_func in ['mean','ptp']:
#                     feats[f"metric_pod_timestamp_value_{stats_func}_{container}_{pod}"] = df_temp.agg(stats_func)
                    
    
        for (metric_name,container,service_name),df in list(metric.groupby(['metric_name','container','service_name'])):
            if len(df)>0:
                feats[f'metric_mci_name_cnt_{metric_name}_{container}_{service_name}'] = len(df)
                for stats_func in ['mean','ptp']:
                    feats[f"metric_mci_name_timestamp_value_{stats_func}_{metric_name}_{container}_{service_name}"] = df['value'].agg(stats_func)  
    
    
        for (metric_name,container,instance),df in list(metric.groupby(['metric_name','container','instance'])):
            if len(df)>0:
                feats[f'metric_msi_name_cnt_{metric_name}_{container}_{instance}'] = len(df)
                for stats_func in ['mean','ptp']:
                    feats[f"metric_msi_name_timestamp_value_{stats_func}_{metric_name}_{container}_{instance}"] = df['value'].agg(stats_func)
                  
        for container,df in list(metric.groupby('container')):
            if len(df)>0:
                feats[f'metric_container_cnt_{container}'] = len(df)
                df_temp=df.groupby(['metric_name'])['value'].mean()#,'timestamp'
                
                for stats_func in ['mean','ptp']:
                    feats[f"metric_timestamp_value_{stats_func}_{container}"] = df_temp.agg(stats_func)
                    
#                 df['diff']=df['timestamp'].diff()
#                 for stats_func in ['mean','std', 'skew', 'kurt','ptp']:
#                     feats[f"metric_diff_{stats_func}_{container}"] =df['diff'].apply(stats_func)

        for service_name,df in list(metric.groupby('service_name')):
            if len(df)>0:
                feats[f'metric_container_service_cnt_{service_name}'] = len(df)
                df_temp=df.groupby(['metric_name'])['value'].mean()#,'timestamp'
                
                for stats_func in ['mean','ptp']:
                    feats[f"metric_service_timestamp_value_{stats_func}_{service_name}"] = df_temp.agg(stats_func)


    else:
        feats['metric_length'] = -1

    return feats
def gen_label(train):
    col = np.zeros((train.shape[0], 24))
    for i, label in enumerate(train['label'].values):
        col[i][label] = 1
    return col
all_ids = set([i.split("_")[0] for i in os.listdir("../model/tmp_data/inputs/metric/")]) |\
          set([i.split("_")[0] for i in os.listdir("../model/tmp_data/inputs/log/")]) |\
          set([i.split("_")[0] for i in os.listdir("../model/tmp_data/inputs/trace/")])
all_ids = list(all_ids)
if ".ipynb" in all_ids:
    all_ids.remove(".ipynb")
print("IDs Length =", len(all_ids))
feature = pd.DataFrame(Parallel(n_jobs=60, backend="multiprocessing")(delayed(processing_feature)(f) for f in tqdm(all_ids)))
# feature.to_csv("new_ids_fuxian.csv", index=None)

feature2 = pd.read_parquet("../other-file-or-dir/model3_list.parquet")
# feature2 = feature2[["id"]]
feature = feature2.merge(feature,on="id",how="left")
col = pd.read_pickle("../other-file-or-dir/model3_col.pickle")
feature = feature.reindex(columns=col)
# feat2 = pd.read_parquet('feat_tags.pqt')
# feature = feature.merge(feat2,on="id",how="left")


# drop_col = pd.read_pickle("drop_yb.pickle")
# feature = feature.drop(columns=drop_col)
print("==> label encoder start")
label = pd.read_csv("../model/tmp_data/labels/training_label.csv")
lb_encoder = LabelEncoder()
label['label'] = lb_encoder.fit_transform(label['source'])
all_data = feature.merge(label[['id', 'label']].groupby(['id'], as_index=False)['label'].agg(list), how='left', on=['id']).set_index("id")
all_data = all_data.sort_values('id')
# host_df = pd.read_pickle("feature/host_ip_max_id_id_host_ip_max_id_b_deepwalk_32.pkl")
# all_data = all_data.merge(host_df,on="host_ip_max_id",how="left").set_index(all_data.index)
# id_host = pd.read_pickle("feature/id_host_ip_max_id_id_b_deepwalk_32.pkl")
# all_data = all_data.merge(id_host,on="id",how="left").set_index(all_data.index)
# for i in ["host_ip","service_name","span_id","trace_id","parent_id", "service"]:#, "service","endpoint_name""trace_id",
#     lb = LabelEncoder()
#     all_data[f"{i}_max_id"] = lb.fit_transform(all_data[f"{i}_max_id"])

not_use = ['id', 'label', 'diff']
feature_name = [i for i in all_data.columns if i not in not_use]
for i in feature_name:
    if all_data[i].dtype == object:
        print(all_data[i].dtype,i)
X = all_data[feature_name].replace([np.inf, -np.inf], 0).clip(-1e9, 1e9)
print(f"Feature Length = {len(feature_name)}")
# print(f"Feature = {feature_name}")
num_classes = 24
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, random_state=2048, shuffle=True)
scaler = StandardScaler()
# scaler_X = scaler.fit_transform(X.fillna(0).replace([np.inf, -np.inf], 0))
scaler_X = scaler.fit_transform(X.fillna(0).replace([np.inf, -np.inf], 0))

# for i in range(len(lb_encoder.classes_)):

y = gen_label(all_data[all_data['label'].notnull()])
train_scaler_X = scaler_X[all_data['label'].notnull()]
test_scaler_X = scaler_X[all_data['label'].isnull()]

ovr_oof = np.zeros(len(train_scaler_X))
result = []
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
all_score = []
for i in range(len(lb_encoder.classes_)):
    print("*"*20, lb_encoder.classes_[i], "*"*20)
#     print(y)
    now_y = y[:, i]
#     print(now_y)
    fold_score = []
    fold_num = 1
    ovr_preds = np.zeros(len(test_scaler_X))
    feature_importance_df = pd.DataFrame()
    for train_index, valid_index in kf.split(train_scaler_X, now_y):
        X_train, X_valid = train_scaler_X[train_index], train_scaler_X[valid_index]
        y_train, y_valid = now_y[train_index], now_y[valid_index]
        clf = XGBClassifier(random_state=2023, n_jobs=62,learning_rate=0.05,n_estimators=500,max_depth=8)
        clf.fit(X_train, y_train,eval_set=[(X_valid, y_valid)],
            eval_metric='auc',
#             early_stopping_rounds=100,
            verbose=50,)
        ovr_oof[valid_index] = clf.predict_proba(X_valid)[:,1]
        # print(clf.predict_proba(test_scaler_X)[:,1] )
        ovr_preds += clf.predict_proba(test_scaler_X)[:,1] / n_splits
#         clf.save_model(f'output2/xgb_{lb_encoder.classes_[i]}_fold_{fold_num}.bin')
        # print(ovr_preds)
        # score = sScore(y_valid, ovr_oof[valid_index])
        Score = roc_auc_score(y_valid, ovr_oof[valid_index])
        fold_score.append(Score)
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feature_name
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = fold_num + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        # print(f"Score = {np.mean(score)}")
        # print("Fold", fold_num,"AUC", Score)
        fold_num = fold_num + 1
    result.append(ovr_preds)
    # each_score = sScore(y, ovr_oof)
    print(f"{lb_encoder.classes_[i]} Mean Score = {np.mean(fold_score)}")
    all_score.append(fold_score)
#     display_importances(feature_importance_df,lb_encoder.classes_[i])
print(f"All Mean Score = {np.mean(all_score)}")

# score_metric = pd.DataFrame(each_score, columns=['score'], index=list(lb_encoder.classes_))
# score_metric.loc["Weighted AVG.", "score"] = np.mean(score_metric['score'])
# print(score_metric)
result = np.array(result).T
submit = pd.DataFrame(result, columns=lb_encoder.classes_)
submit.index = all_data[all_data['label'].isnull()].index
submit.reset_index(inplace=True)
submit = submit.melt(id_vars="id", value_vars=lb_encoder.classes_, value_name="score", var_name="source")
submit.to_csv(f"../model/tmp_data/model_3.csv", index=False)