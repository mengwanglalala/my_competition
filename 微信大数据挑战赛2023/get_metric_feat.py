import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


from tqdm import tqdm
import multiprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import Parallel, delayed

path = '../model/tmp_data/inputs/'

error_service=['15b7f6577fec4c16b01ee2e053b1f201',
'6cc7dc7bb5fa4327a20c883ab00ab2fe',
'00c1ba198361424c9597328ea33d0d15',
'c453a975de5148e4b1c47be258a646c9',
'faf90b12d1cf478e810172eb6aced658',
'05be78908e3c4818b1caa00b71d8bb11',
'03d1f58da52d49dbb815cda9be061d25',
'2170e75abdf54178afcd5ffecb387eee',
'0d1304f1f40743dea03be55bca96c32b',
'14eb4630112b4ce9bd88d93104b4570e',
'8b18231981e0440488bbac370b1464cf',
'36c4ac32f7504f13b7aef941de9ecc81',
'53f1acb37db941b8b9c77dfefecb157b',
'8b3eee3cc4fe4568b5ba4125c1a4047f',
'122ec12af3744773b9b04c6c8e929711',
'3d82f4ad7f114cbdbd469fc897b001a1',
'e97a387ed0204878b0660f0090bfacd6',
'4287f5cca47742008a8fb965908e5dea',
'f1023ca9976e4a5eaaaaed244acd2f4a',
'ea4bdf00441c4157a99a9c72bb7f4eb2',
'node-worker1',
'node-worker2',
'node-worker3']


import json

def extract_metric_name(json_str):
    # 将JSON格式的字符串解析为Python字典
    data = json.loads(json_str)

    # 提取"metric_name"字段的值
    metric_name = data.get("metric_name")

    return metric_name


def extract_service_name(json_str):
    # 将JSON格式的字符串解析为Python字典
    data = json.loads(json_str)

    # 提取"metric_name"字段的值
    metric_name = data.get("service_name")

    return metric_name

def extract_container(json_str):
    # 将JSON格式的字符串解析为Python字典
    data = json.loads(json_str)

    # 提取"metric_name"字段的值
    metric_name = data.get("container")

    return metric_name

def extract_instance(json_str):
    # 将JSON格式的字符串解析为Python字典
    data = json.loads(json_str)

    # 提取"metric_name"字段的值
    metric_name = data.get("instance")

    return metric_name

def extract_kubernetes_io_hostname(json_str):
    # 将JSON格式的字符串解析为Python字典
    data = json.loads(json_str)

    # 提取"metric_name"字段的值
    metric_name = data.get("kubernetes_io_hostname")

    return metric_name


def extract_namespace(json_str):
    # 将JSON格式的字符串解析为Python字典
    data = json.loads(json_str)

    # 提取"metric_name"字段的值
    metric_name = data.get("namespace")

    return metric_name

def extract_mode(json_str):
    # 将JSON格式的字符串解析为Python字典
    data = json.loads(json_str)

    # 提取"metric_name"字段的值
    metric_name = data.get("mode")

    return metric_name

def extract_cpu(json_str):
    # 将JSON格式的字符串解析为Python字典
    data = json.loads(json_str)

    # 提取"metric_name"字段的值
    metric_name = data.get("cpu")

    return metric_name

def extract_interface(json_str):
    # 将JSON格式的字符串解析为Python字典
    data = json.loads(json_str)

    # 提取"metric_name"字段的值
    metric_name = data.get("interface")

    return metric_name

def extract_pod(json_str):
    # 将JSON格式的字符串解析为Python字典
    data = json.loads(json_str)

    # 提取"metric_name"字段的值
    metric_name = data.get("pod")

    return metric_name

def extract_job(json_str):
    # 将JSON格式的字符串解析为Python字典
    data = json.loads(json_str)

    # 提取"metric_name"字段的值
    metric_name = data.get("job")

    return metric_name


name_list=['container_cpu_system_seconds_total',' container_cpu_usage_seconds_total',' container_cpu_user_seconds_total', 'cpm',
       'resp_time', 'error_count', 'success_rate']

name_list_2=['container_network_receive_bytes_total','container_network_receive_errors_total','container_network_receive_packets_dropped_total', 'container_network_receive_packets_total',
       'container_network_transmit_bytes_total', 'container_network_transmit_errors_total', 'container_network_transmit_packets_dropped_total','container_network_transmit_packets_total']

instance_list=['10.60.252.14:9100', '10.60.61.196:9100', '10.60.7.88:9100']


##baseline  19个
def processing_feature(file):
    metric = pd.DataFrame()
    if os.path.exists(path + f"metric/{file}_metric.csv"):
        metric = pd.read_csv(path + f"metric/{file}_metric.csv").sort_values(by=['timestamp']).reset_index(
            drop=True)

    feats = {"id": file}

    if len(metric) > 0:
        feats['metric_length'] = len(metric)
        df_temp = metric.groupby(['timestamp'])['value'].mean()
        for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
            feats[f"metric_value_timestamp_value_{stats_func}"] = df_temp.apply(stats_func)

        metric['diff'] = metric['timestamp'].diff()
        for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
            feats[f"metric_diff_{stats_func}"] = metric['diff'].apply(stats_func)

        feats[f"metric_value_timestamp_value_max_max"] = metric['value'].max()
        feats[f"metric_value_timestamp_value_min_min"] = metric['value'].min()
        feats[f"metric_value_timestamp_value_std_max"] = metric.groupby(['timestamp'])['value'].std().max()
        feats[f"metric_value_timestamp_value_std_min"] = metric.groupby(['timestamp'])['value'].std().min()

        metric['metric_name'] = metric['tags'].apply(extract_metric_name)
        metric['hostname'] = metric['tags'].apply(extract_kubernetes_io_hostname)
        metric['container'] = metric['tags'].apply(extract_container)
        metric['service_name'] = metric['tags'].apply(extract_service_name)
        metric['namespace'] = metric['tags'].apply(extract_namespace)
        metric['instance'] = metric['tags'].apply(extract_instance)
        metric['mode'] = metric['tags'].apply(extract_mode)
        metric['cpu'] = metric['tags'].apply(extract_cpu)
        metric['interface'] = metric['tags'].apply(extract_interface)
        metric['job'] = metric['tags'].apply(extract_job)
        metric['pod'] = metric['tags'].apply(extract_pod)

        feats['metric_str_select_cnt_metric_name'] = metric['metric_name'].nunique()
        feats['metric_str_select_cnt_service_name'] = metric['service_name'].nunique()
        feats['metric_str_select_cnt_container'] = metric['container'].nunique()

        for metric_name, df in list(metric.groupby('metric_name')):
            if len(df) > 0:
                feats[f'metric_name_cnt_{metric_name}'] = len(df)
                df_temp = df.groupby(['timestamp'])['value'].mean()

                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                    feats[f"metric_timestamp_value_{stats_func}_{metric_name}"] = df_temp.apply(stats_func)

        for hostname, df in list(metric.groupby('hostname')):
            if len(df) > 0:
                feats[f'metric_hostname_cnt_{hostname}'] = len(df)
                df_temp = df.groupby(['timestamp'])['value'].mean()

                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                    feats[f"metric_timestamp_value_{stats_func}_{hostname}"] = df_temp.apply(stats_func)

        for container, df in list(metric.groupby('container')):
            if len(df) > 0:
                feats[f'metric_container_cnt_{container}'] = len(df)
                df_temp = df.groupby(['timestamp'])['value'].mean()

                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                    feats[f"metric_timestamp_value_{stats_func}_{container}"] = df_temp.apply(stats_func)

        for service_name, df in list(metric.groupby('service_name')):
            if len(df) > 0:
                feats[f'metric_service_name_cnt_{service_name}'] = len(df)
                df_temp = df.groupby(['timestamp'])['value'].mean()

                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                    feats[f"metric_service_name_timestamp_value_{stats_func}_{service_name}"] = df_temp.apply(
                        stats_func)

                feats[f"metric_value_timestamp_value_max_max_service_{service_name}"] = df['value'].max()
                feats[f"metric_value_timestamp_value_min_min_service_{service_name}"] = df['value'].min()
                feats[f"metric_value_timestamp_value_std_max_service_{service_name}"] = df.groupby(['timestamp'])[
                    'value'].std().max()
                feats[f"metric_value_timestamp_value_std_min_service_{service_name}"] = df.groupby(['timestamp'])[
                    'value'].std().min()

                df['diff'] = df['timestamp'].diff()
                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                    feats[f"metric_service_name_diff_{stats_func}_{service_name}"] = df['diff'].apply(stats_func)

        for (metric_name, service_name), df in metric.groupby(['metric_name', 'service_name']):
            if len(df) > 0:
                feats[f'metric_service_name_cnt_{service_name}_{metric_name}'] = len(df)

                feats[f"metric_value_timestamp_value_kurt_service_{service_name}_{metric_name}"] = df['value'].agg(
                    'kurt')
                feats[f"metric_value_timestamp_value_skew_service_{service_name}_{metric_name}"] = df['value'].agg(
                    'skew')

                feats[f"metric_value_timestamp_value_mean_service_{service_name}_{metric_name}"] = df['value'].mean()
                feats[f"metric_value_timestamp_value_max_service_{service_name}_{metric_name}"] = df['value'].max()
                feats[f"metric_value_timestamp_value_min_service_{service_name}_{metric_name}"] = df['value'].min()
                feats[f"metric_value_timestamp_value_std_service_{service_name}_{metric_name}"] = df['value'].std()

        for (metric_name, container, instance), df in list(metric.groupby(['metric_name', 'container', 'instance'])):
            if len(df) > 0:
                feats[f'metric_mci_name_cnt_{metric_name}_{container}_{instance}'] = len(df)

                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                    feats[f"metric_mci_name_timestamp_value_{stats_func}_{metric_name}_{container}_{instance}"] = df[
                        'value'].agg(stats_func)

        for (metric_name, container, job), df in list(metric.groupby(['metric_name', 'container', 'job'])):
            if len(df) > 0:
                feats[f'metric_mcj_name_cnt_{metric_name}_{container}_{job}'] = len(df)
                df_temp = df.groupby(['timestamp'])['value'].agg('mean')
                for stats_func in ['mean', 'ptp']:
                    feats[
                        f"metric_mcj_name_timestamp_value_{stats_func}_{metric_name}_{container}_{job}"] = df_temp.agg(
                        stats_func)

        metric_chao = metric[metric['container'] == 'chaosblade-tool']
        feats[f'metric_chao_metric_name_cnt'] = metric_chao['metric_name'].nunique()

        for metric_name, df in list(metric_chao.groupby('metric_name')):
            if len(df) > 0:
                feats[f'metric_chao_name_cnt_{metric_name}'] = len(df)
                df_temp = df.groupby(['timestamp'])['value'].mean()

                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                    feats[f"metric_chao_name_timestamp_value_{stats_func}_{metric_name}"] = df_temp.apply(stats_func)

        for i in name_list:
            metric_temp = metric[metric['metric_name'] == i]

            for j in ['namespace', 'instance', 'mode', 'cpu']:
                for name, df in list(metric_temp.groupby(j)):
                    if len(df) > 0:
                        feats[f'metric_{j}_name_cnt_{name}_{i}'] = len(df)

                        df_temp = df.groupby(['timestamp'])['value'].mean()

                        for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                            feats[f"metric_{j}_name_timestamp_value_{stats_func}_{name}_{i}"] = df_temp.apply(
                                stats_func)

        metric_ = metric[metric['instance'].isin(instance_list)]

        for instance in instance_list:
            metric_temp = metric_[metric_['instance'] == instance]
            for (metric_name, job, cpu, mode), df in list(metric_temp.groupby(['metric_name', 'job', 'cpu', 'mode'])):
                if len(df) > 0:
                    feats[f'metric_imjcm_cnt_{instance}_{metric_name}_{job}_{cpu}_{mode}'] = len(df)

                    for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                        feats[f"metric_imjcm_value_{stats_func}_{instance}_{metric_name}_{job}_{cpu}_{mode}"] = df[
                            'value'].agg(stats_func)

        metric_ = metric[metric['container'] == '']
        metric_ = metric_[metric_['job'] == 'kubernetes-nodes-cadvisor']

        metric_['pod_new'] = metric_['pod'].apply(lambda x: str(x).split('-')[0])

        for (metric_name, pod), df in list(metric_.groupby(['metric_name', 'pod_new'])):
            if len(df) > 0:
                feats[f'metric_mp_name_cnt_{metric_name}_{pod}'] = len(df)

                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                    feats[f"metric_mp_name_timestamp_value_{stats_func}_{metric_name}_{pod}"] = df['value'].agg(
                        stats_func)

        metric = metric.drop(['diff'], axis=1)
        metric = metric.drop_duplicates().reset_index(drop=True)

        for i in name_list_2:
            metric_ = metric[metric['metric_name'] == i]

            for (metric_name, container, instance, interface), df in list(
                    metric_.groupby(['metric_name', 'container', 'instance', 'interface'])):
                if len(df) > 0:
                    if df['timestamp'].nunique() == len(df):
                        feats[f'metric_mci_name_cnt_{metric_name}_{container}_{instance}_{interface}'] = len(df)

                        for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                            feats[
                                f"metric_mcii_name_timestamp_value_{stats_func}_{metric_name}_{container}_{instance}_{interface}"] = \
                            df['value'].agg(stats_func)

        metric_chao = metric[metric['container'] == 'chaosblade-tool']
        for tags, df in list(metric_chao.groupby(['tags'])):
            if len(df) > 0:

                feats[f'metric_chaosb_name_cnt_{tags}'] = len(df)

                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                    feats[f"metric_chaosb_name_timestamp_value_{stats_func}_{tags}"] = df['value'].agg(stats_func)


    else:
        feats['metric_length'] = -1

    return feats


all_ids = set([i.split("_")[0] for i in os.listdir(path+"metric/") ]) |\
          set([i.split("_")[0] for i in os.listdir(path+"log/")   ]) |\
          set([i.split("_")[0] for i in os.listdir(path+"trace/")  ])
all_ids = list(all_ids)
print("IDs Length =", len(all_ids))

if ".ipynb" in all_ids:
    all_ids.remove(".ipynb")
    

all_ids=sorted(all_ids)

data = pd.DataFrame(Parallel(n_jobs=60, backend="multiprocessing")(delayed(processing_feature)(f) for f in tqdm(all_ids)))

import pickle
with open(f'../other-file-or-dir/tmp_pickle/feat13_dict.pickle', 'rb') as file:
    new_column_names = pickle.load(file)
    
data.rename(columns=new_column_names, inplace=True)

data.to_parquet('../model/tmp_data/metric_feat.pqt')