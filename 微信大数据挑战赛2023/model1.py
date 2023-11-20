import os
import warnings

warnings.filterwarnings('ignore')
import random
import numpy as np
import pandas as pd
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


random.seed(2021)
np.random.seed(2021)
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
    if os.path.exists(f"../model/tmp_data/inputs/log/{file}_log.csv"):
        log = pd.read_csv(f"../model/tmp_data/inputs/log/{file}_log.csv").sort_values(by=['timestamp']).reset_index(drop=True)

    if os.path.exists(f"../model/tmp_data/inputs/trace/{file}_trace.csv"):
        trace = pd.read_csv(f"../model/tmp_data/inputs/trace/{file}_trace.csv").sort_values(by=['timestamp']).reset_index(drop=True)

    if os.path.exists(f"../model/tmp_data/inputs/metric/{file}_metric.csv"):
        metric = pd.read_csv(f"../model/tmp_data/inputs/metric/{file}_metric.csv").sort_values(by=['timestamp']).reset_index(drop=True)

          
        
    def compute_features(df):
        if len(df) > 0:
            service = df['service'].iloc[0]  # 获取分组的 'service' 值
            feats[f'log_service_cnt_{service}'] = len(df)
            df['diff'] = df['timestamp'].diff()
            stats_funcs = ['mean', 'std', 'skew', 'kurt', 'ptp', 'max', 'min']
            for stats_func in stats_funcs:
                feats[f"log_time_diff_{stats_func}_{service}"] = df['diff'].agg(stats_func)

            
    def service_groupby(df):
        if len(df)>0:
            service = df['service_name'].iloc[0]
            feats[f'trace_service_name_cnt_{service}'] = len(df)
            df['diff'] = df['timestamp'].diff()
            stats_funcs = ['mean', 'std', 'skew', 'kurt', 'ptp', 'max', 'min']
            for stats_func in stats_funcs:
                feats[f"log_time_diff_{stats_func}_{service}"] = df['diff'].agg(stats_func)        
        
        
        
        
        
    feats = {"id": file}
    if len(trace) > 0:
        feats['trace_length'] = len(trace)
        feats[f"trace_status_code_std"] = trace['status_code'].apply("std")
        
        
        trace.groupby("service_name").apply(service_groupby)

        for stats_func in ['std','skew',"kurt"]:#'mean',, 'nunique''std', 
            feats[f"trace_timestamp_{stats_func}"] = trace['timestamp'].apply(stats_func)

        for stats_func in ['nunique']:
            for i in ['host_ip', 'service_name', 'trace_id', 'parent_id','span_id']:#, 'start_time','end_time' , 'endpoint_name', 
                feats[f"trace_{i}_{stats_func}"] = trace[i].agg(stats_func)
        trace["timestamp_diff"] = trace["timestamp"].diff(1)
#         feats["timestamp_diff"] = trace["timestamp"].max() - trace["timestamp"].min()
#         trace["timestamp_diff"] = trace["timestamp"].max() - trace["timestamp"].min()
#         feats["timestamp_diff_sum"] =  trace["timestamp_diff"].sum()
#         feats["timestamp_diff_max"] = trace["timestamp_diff"].max()
#         feats["timestamp_diff_std"] =  trace["timestamp_diff"].std()
#         feats["timestamp_diff_mean"] = trace["timestamp_diff"].mean()
        trace["timestamp_diff"] = trace["timestamp"].diff(1)
        feats["timestamp_diff"] = trace["timestamp"].max() - trace["timestamp"].min()
        feats["timestamp_diff_sum"] =  trace["timestamp_diff"].sum()
        feats["timestamp_diff_max"] = trace["timestamp_diff"].max()
        feats["timestamp_diff_std"] =  trace["timestamp_diff"].std()
        feats["timestamp_diff_mean"] = trace["timestamp_diff"].mean()
        feats['autocorr_start'] = trace['start_time'].autocorr()
        feats['autocorr_end'] = trace['end_time'].autocorr()
#         trace['start_time'] = pd.to_datetime(trace['start_time'])  # Convert to datetime type
#         trace['end_time'] = pd.to_datetime(trace['end_time'])
        feats['diff'] = trace["end_time"] - trace["start_time"]
        trace['diff'] = trace["end_time"] - trace["start_time"]
        feats['timestamp_count'] = trace[(trace['timestamp'] < 100) | (trace['timestamp'] == 1199999)]['timestamp'].count()
        mysql_data = trace[trace['endpoint_name'].str.contains('mysql', case=False)]
        feats['diff'] = mysql_data['end_time'] - mysql_data['start_time']
        trace['diff'] = mysql_data['end_time'] - mysql_data['start_time']
        feats[f'trace_mysql_mean'] = (mysql_data['end_time'] - mysql_data['start_time']).mean()
        feats[f'trace_mysql_std'] = (mysql_data['end_time'] - mysql_data['start_time']).std()
        feats[f'trace_mysql_max'] = (mysql_data['end_time'] - mysql_data['start_time']).max()
        feats[f'trace_mysql_nunique'] = (mysql_data['end_time'] - mysql_data['start_time']).nunique()
        feats[f'trace_mysql_min'] = (mysql_data['end_time'] - mysql_data['start_time']).min()
        
        
#         feats[f'trace_mysql_mean'] = trace['diff'].str.contains(text, case=False).mean()
        #pd.to_datetime(trace["end_time"]) - pd.to_datetime(trace["start_time"])
#         feats['diff'] = pd.to_datetime(trace["end_time"]) - pd.to_datetime(trace["start_time"])
        trace['duration'] = trace["end_time"] - trace["start_time"]#pd.to_datetime(trace["end_time"]) - pd.to_datetime(trace["start_time"])
#         trace['duration'] =  mysql_data['end_time'] - mysql_data['start_time']#pd.to_datetime(trace["end_time"]) - pd.to_datetime(trace["start_time"])
    
        feats['diff_mean'] = feats['diff'].mean()#.total_seconds()
        feats['diff_std'] = feats['diff'].std()#.total_seconds()
        
        feats['diff_ptp'] = feats['diff'].max() - feats['diff'].min()#feats['diff'].dt.total_seconds().max() - feats['diff'].dt.total_seconds().min()
#         feats['duration_min'] = trace['duration'].min().total_seconds()

        feats['duration_max'] = trace['diff'].max()#.total_seconds()
        
        for i in ["host_ip","service_name","trace_id","span_id","parent_id"]:#,"endpoint_name"
            feats[f"{i}_max_id"] = trace[i].value_counts().reset_index().sort_values([i, "count"], ascending=[False, True]).reset_index(drop=True)[:1][i].values[0]
            trace[f"{i}_max_id"] = trace[i].value_counts().reset_index().sort_values([i, "count"], ascending=[False, True]).reset_index(drop=True)[:1][i].values[0]
            
            ids = trace[i].value_counts().reset_index().sort_values([i, "count"], ascending=[False, True]).reset_index(drop=True)[:1][i].values[0]
#             feats[f"{i}_max_id_diff_max"] = strace.loc[trace[i] == ids, "diff"].values[0]
            feats[f"{i}_max_id_diff_max"] = trace[trace[i] ==  ids]['diff'].max()#.dt.total_seconds().max() , ascending=[False, True]
            feats[f"{i}_max_id_diff_mean"] = trace[trace[i] ==  ids]['diff'].mean()
            feats[f"{i}_max_id_diff_std"] = trace[trace[i] ==  ids]['diff'].std()
#             feats[f"{i}_max_id_diff_ptp"] = trace[trace[i] ==  ids]['diff'].max() - trace[trace[i] ==  ids]['diff'].min()

            #         grouped = trace.groupby('trace_id')
#         trace['error_rit'] = grouped['timestamp'].diff()
#         trace.loc[trace['status_code'] == 200, 'error_rit'] = 0
#         feats['error_rit'] = trace['error_rit'].sum()
#         feats['diff'] = feats['diff'].dt.total_seconds()
        
        for percentile in [25, 75]:
            feats[f"duration_{percentile}th_percentile"] = trace['duration'].quantile(percentile / 100)
# #         feats['duration_variation'] = trace['duration'].std().total_seconds()
# #         trace['start_hour'] = trace['start_time'].dt.hour
# #         duration_hourly_mean = trace.groupby('start_hour')['duration'].mean()
# #         feats['duration_hourly_mean_std'] = duration_hourly_mean.std().total_seconds()
# #         trace['end_time'] = pd.to_datetime(trace['end_time'])
        trace['next_start_time'] = trace['start_time'].shift(-1)  # 将start_time列向后移动以获取下一个start_time
        trace['next_start_time'] = trace['next_start_time']
        feats['interval_mean'] = (trace['next_start_time'] - trace['end_time']).mean()
        feats['interval_std'] = (trace['next_start_time'] - trace['end_time']).std()
#             feats[f"{i}_max_counts"] = trace[i].value_counts().max()
            
#         feats['autocorr_timestamp'] = trace['timestamp'].autocorr()
#         feats['autocorr_diff'] = trace['diff'].autocorr()
#         fft_result = np.fft.fft(trace['diff'])
#         trace['fft_result'] = fft_result
#         frequency_spectrum = np.abs(fft_result)
#         frequency_bins = np.fft.fftfreq(len(trace['diff'])) 
#         feats['frequency_spectrum_mean'] = np.mean(frequency_spectrum)
#         feats['frequency_spectrum_max'] = np.max(frequency_spectrum)
#         feats['frequency_spectrum_min'] = np.min(frequency_spectrum)
#         feats['frequency_spectrum_var'] = np.var(frequency_spectrum)
#         feats['peak_frequency'] = frequency_bins[np.argmax(frequency_spectrum)]
#         feats['energy'] = np.sum(frequency_spectrum)
#         feats['percentile_25th'] = np.percentile(frequency_spectrum, 25)
#         feats['percentile_75th'] = np.percentile(frequency_spectrum, 75)


#         angle = np.angle(fft_result)
#         # 计算相位角的统计特征
#         feats['angle_mean'] = np.mean(angle)  # 平均值
#         feats['angle_std'] = np.std(angle)  # 标准差
#         feats['angle_max'] = np.max(angle)  # 最大值
#         feats['angle_min'] = np.min(angle)  # 最小值
        
        
        for i in ["host_ip"]:
            feats[f'trace_groupby_{i}_diff_nunique_mean'] = trace.groupby([i])['diff'].nunique().mean()
            feats[f'trace_groupby_{i}_diff_nunique_std'] = trace.groupby([i])['diff'].nunique().std()
            feats[f'trace_groupby_{i}_diff_mean_std'] = trace.groupby([i])['diff'].mean().std()
            feats[f'trace_groupby_{i}_diff_max'] = trace.groupby([i])['diff'].sum().max()
            feats[f'trace_groupby_{i}_diff_max'] = trace.groupby([i])['diff'].sum().min()

#             feats[f'trace_groupby_{i}_diff_ptp'] = trace.groupby([i])['diff'].dt.total_seconds().max() - trace.groupby([i])['diff'].dt.total_seconds().min()
#         统计操作类型组合的次数
#         combinations = trace.groupby(['endpoint_name']).apply(lambda x: '->'.join(x['endpoint_name'].shift().fillna('')))
#         combination_counts = combinations.value_counts().reset_index()
#         combination_counts.columns = ['combination', 'count']
#         feats['combination_counts_sum'] = combination_counts['count'].sum()
#         feats['combination_counts_mean'] = combination_counts['count'].mean()
        

#         统计操作类型转换的次数
#         transitions = trace.groupby(['endpoint_name']).apply(lambda x: '->'.join(x['endpoint_name'].shift(-1).fillna('')))
#         transition_counts = transitions.value_counts().reset_index()
#         transition_counts.columns = ['transition', 'count']
#         feats['transition_counts_sum'] = transition_counts['count'].sum()
#         feats['transition_counts_mean'] = transition_counts['count'].mean()
#         feats['diff'] = trace['diff'].dt.total_seconds()
        
        
        

    else:
        feats['trace_length'] = -1

    if len(log) > 0:
        feats['log_length'] = len(log)
        log['time_diff'] = log['timestamp'].diff(1)#.shift(-1) - log['timestamp']
        log.loc[log.index[-1], 'time_diff'] = 0
        grouped = log.groupby("service")["time_diff"]
        feats['log_mean_time_diff'] = grouped.mean().to_numpy()[0]
        feats['log_std_time_diff'] = grouped.std().to_numpy()[0]
        feats['log_min_time_diff'] = grouped.min().to_numpy()[0]
        feats['log_max_time_diff'] = grouped.max().to_numpy()[0]
#         feats['log_ptp_time_diff'] = grouped.max().to_numpy()[0] - grouped.min().to_numpy()[0]
        
        log['message_length'] = log['message'].fillna("").map(len)
        log['log_info_length'] = log['message'].map(lambda x: x.split("INFO")).map(len)
        feats['service_max_id'] = log["service"].value_counts().reset_index().sort_values(["service","count"], ascending=[False, True]).reset_index(drop=True)[:1]["service"].values[0]#, ascending=[False, True]
        feats['log_service_nunique'] = log['service'].nunique()
        feats['message_length_std'] = log['message'].fillna("").map(len).std()
        feats['message_length_ptp'] = log['message'].fillna("").map(len).agg('ptp')
        feats['log_info_length'] = log['message'].map(lambda x:x.split("INFO")).map(len).agg('ptp')
        log.groupby('service').apply(compute_features)
            
        #developed feature
        # text_list = ['异常','错误','error','user','mysql','true','失败']
#         text_list = ['user','mysql']
#         for text in text_list:
#             feats[f'message_{text}_sum'] = log['message'].str.contains(text, case=False).sum()
#             feats[f'message_{text}_mean'] = log['message'].str.contains(text, case=False).mean()
#         feats[f'message_mysql_mean'] = 1 if feats[f'message_mysql_mean'] > 0 else 0

    else:
        feats['log_length'] = -1

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
                    
    
#         for (metric_name,container,service_name),df in list(metric.groupby(['metric_name','container','service_name'])):
#             if len(df)>0:
#                 feats[f'metric_mci_name_cnt_{metric_name}_{container}_{service_name}'] = len(df)
#                 for stats_func in ['mean','ptp']:
#                     feats[f"metric_mci_name_timestamp_value_{stats_func}_{metric_name}_{container}_{service_name}"] = df['value'].agg(stats_func)
    
    
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

        
# #         for hostname,df in list(metric.groupby('hostname')):
# #             if len(df)>0:
# #                 feats[f'metric_hostname_cnt_{hostname}'] = len(df)
# #                 df_temp=df.groupby(['metric_name','timestamp'])['value'].mean()
                
# #                 for stats_func in ['mean','std', 'skew', 'kurt','ptp']:
# #                     feats[f"metric_timestamp_value_{stats_func}_{hostname}"] = df_temp.apply(stats_func)
                    
# #                 df['diff']=df['timestamp'].diff()
# #                 for stats_func in ['mean','std', 'skew', 'kurt','ptp']:
# #                     feats[f"metric_diff_{stats_func}_{hostname}"] =df['diff'].apply(stats_func)
                    
# #         for container,df in list(metric.groupby(['container','timestamp'])):
# #             if len(df)>0:
# #                 feats[f'metric_container_cnt_{container}'] = len(df)
# #                 #f_temp=df.groupby(['metric_name','timestamp'])['value'].mean()
                
# #                 for stats_func in ['mean','ptp']:
# #                     feats[f"metric_timestamp_value_{stats_func}_{container}"] = df['value'].agg(stats_func)
                    
# #                 df['diff']=df['timestamp'].diff()
# #                 for stats_func in ['mean','std', 'skew', 'kurt','ptp']:
# #                     feats[f"metric_diff_{stats_func}_{container}"] =df['diff'].apply(stats_func)

#         for service_name,df in list(metric.groupby(['service_name','timestamp','metric_name'])):
#             if len(df)>0:
#                 feats[f'metric_container_cnt_{service_name}'] = len(df)
# #                 df_temp=df.groupby(['metric_name','timestamp'])['value']#.mean()
                
#                 for stats_func in ['mean','ptp']:#'std', 'skew', 'kurt',
#                     feats[f"metric_service_timestamp_value_{stats_func}_{service_name}"] = df['value'].agg(stats_func)
                    
#                 df['diff']=df['timestamp'].diff()
#                 for stats_func in ['mean','std', 'skew', 'kurt','ptp']:
#                     feats[f"metric_service_diff_{stats_func}_{service_name}"] =df['diff'].apply(stats_func)

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
# feature.to_csv("all_id_fuxian.csv", index=None)
# feature = pd.read_parquet("pod.parquet")
feature2 = pd.read_parquet("../other-file-or-dir/id_list.parquet")
# feature2 = feature2[["id"]]
feature = feature2.merge(feature,on="id",how="left")
col = pd.read_pickle("../other-file-or-dir/model1_col.pickle")
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
for i in ["host_ip","service_name","span_id","trace_id","parent_id", "service"]:#, "service","endpoint_name""trace_id",
    lb = LabelEncoder()
    all_data[f"{i}_max_id"] = lb.fit_transform(all_data[f"{i}_max_id"])
# print(all_data["host_ip_max_id_diff_max"])
# print(all_data["service_name_max_id_diff_max"])
# print(all_data["trace_id_max_id_diff_max"])


# print(all_data["host_ip_max_id_diff_max"])
#     grouped = all_data.groupby(f"{i}_max_id")
    # 将diff列转换为数值类型
#     all_data['diff'] = pd.to_numeric(all_data['diff'], errors='coerce')

    # 按照i_max_id进行分组并计算每个组的diff均值
#     all_data['grouped_diff_sum'] = all_data.groupby(f"{i}_max_id")['diff_mean'].transform('sum')
#     all_data['grouped_diff_std'] = all_data.groupby(f"{i}_max_id")['diff_mean'].transform('std')
# #     all_data['grouped_diff_mean'] = all_data.groupby(f"{i}_max_id")['diff_mean'].transform('mean')
#     all_data['grouped_diff_nunique'] = all_data.groupby(f"{i}_max_id")['diff_mean'].transform('nunique')
#     all_data['grouped_diff_std_sum'] = all_data.groupby(f"{i}_max_id")['diff_std'].transform('sum')
#     all_data['grouped_diff_std_std'] = all_data.groupby(f"{i}_max_id")['diff_std'].transform('std')
#     all_data['grouped_diff_std_mean'] = all_data.groupby(f"{i}_max_id")['diff_std'].transform('mean')
#     all_data['grouped_diff_std_nunique'] = all_data.groupby(f"{i}_max_id")['diff_std'].transform('nunique')
    
#     all_data['grouped_diff_ptp_sum'] = all_data.groupby(f"{i}_max_id")['diff_ptp'].transform('sum')
#     all_data['grouped_diff_ptp_std'] = all_data.groupby(f"{i}_max_id")['diff_ptp'].transform('std')
#     all_data['grouped_diff_ptp_mean'] = all_data.groupby(f"{i}_max_id")['diff_ptp'].transform('mean')
#     all_data['grouped_diff_ptp_nunique'] = all_data.groupby(f"{i}_max_id")['diff_ptp'].transform('nunique')
#     all_data['grouped_diff_ptp_max'] = all_data.groupby(f"{i}_max_id")['diff_ptp'].transform('max')
    
    
#     all_data['grouped_duration_max_sum'] = all_data.groupby(f"{i}_max_id")['duration_max'].transform('sum')
#     all_data['grouped_duration_max_std'] = all_data.groupby(f"{i}_max_id")['duration_max'].transform('std')
#     all_data['grouped_duration_max_mean'] = all_data.groupby(f"{i}_max_id")['duration_max'].transform('mean')
#     all_data['grouped_duration_max_nunique'] = all_data.groupby(f"{i}_max_id")['duration_max'].transform('nunique')
#     all_data['grouped_duration_max_max'] = all_data.groupby(f"{i}_max_id")['duration_max'].transform('max')
    
# #     all_data['grouped_diff_ptp_sum'] = all_data.groupby(f"{i}_max_id")['diff_ptp'].transform('sum')
# #     all_data['grouped_diff_ptp_std'] = all_data.groupby(f"{i}_max_id")['diff_ptp'].transform('std')
# #     all_data['grouped_diff_ptp_mean'] = all_data.groupby(f"{i}_max_id")['diff_ptp'].transform('mean')
# #     all_data['grouped_diff_ptp_nunique'] = all_data.groupby(f"{i}_max_id")['diff_ptp'].transform('nunique')
    
#     all_data['grouped_diff_combination_counts_sum_sum'] = all_data.groupby(f"{i}_max_id")['combination_counts_sum'].transform('sum')
#     all_data['grouped_diff_combination_counts_sum_std'] = all_data.groupby(f"{i}_max_id")['combination_counts_sum'].transform('std')
#     all_data['grouped_diff_combination_counts_sum_mean'] = all_data.groupby(f"{i}_max_id")['combination_counts_sum'].transform('mean')
#     all_data['grouped_diff_combination_counts_sum_nunique'] = all_data.groupby(f"{i}_max_id")['diff_ptp'].transform('nunique')
#     all_data['grouped_diff_transition_counts_sum_sum'] = all_data.groupby(f"{i}_max_id")['transition_counts_sum'].transform('sum')
#     all_data['grouped_diff_transition_counts_sum_std'] = all_data.groupby(f"{i}_max_id")['transition_counts_sum'].transform('std')
#     all_data['grouped_diff_transition_counts_sum_mean'] = all_data.groupby(f"{i}_max_id")['transition_counts_sum'].transform('mean')
#     all_data['grouped_diff_transition_counts_sum_nunique'] = all_data.groupby(f"{i}_max_id")['transition_counts_sum'].transform('nunique')

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
kf = StratifiedKFold(n_splits=n_splits, random_state=3407, shuffle=True)
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
        clf = XGBClassifier(random_state=2021, n_jobs=62,learning_rate=0.05,n_estimators=500)
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
submit.to_csv(f"../model/tmp_data/model_1.csv", index=False)