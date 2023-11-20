import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
import multiprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import Parallel, delayed

path = '../model/tmp_data/inputs/'


def get_endpoint(x):
    if 'GET' in x:
        return 'GET'
    elif 'Mysql' in x:
        return 'Mysql'
    elif 'DELETE' in x:
        return 'DELETE'
    elif 'HikariCP' in x:
        return 'HikariCP'
    elif 'POST' in x:
        return 'POST'
    elif 'api' in x:
        return 'api'




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



end_name=['GET:/api/v1/adminrouteservice/adminroute',
 'GET:/api/v1/verifycode/generate',
 'Mysql/JDBI/Connection/close',
 'POST:/api/v1/adminbasicservice/adminbasic/contacts',
 'POST:/api/v1/adminbasicservice/adminbasic/stations',
 'POST:/api/v1/adminrouteservice/adminroute',
 'POST:/api/v1/admintravelservice/admintravel',
 'POST:/api/v1/adminuserservice/users',
 'POST:/api/v1/users/login']


def calculate_entropy(column):
    value_counts = column.value_counts()
    probabilities = value_counts / len(column)
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy, probabilities.max(), probabilities.min(), probabilities.median()


##baseline  19个
def processing_feature(file):
    log, trace, metric = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if os.path.exists(path + f"trace/{file}_trace.csv"):
        trace = pd.read_csv(path + f"trace/{file}_trace.csv").sort_values(by=['timestamp']).reset_index(
            drop=True)

    feats = {"id": file}
    if len(trace) > 0:

        feats[f"trace_status_code_std"] = trace['status_code'].apply("std")

        trace["timestamp_diff"] = trace["timestamp"].diff(1)
        feats["timestamp_diff"] = trace["timestamp"].max() - trace["timestamp"].min()
        feats["timestamp_diff_sum"] = trace["timestamp_diff"].sum()
        feats["timestamp_diff_max"] = trace["timestamp_diff"].max()
        feats["timestamp_diff_std"] = trace["timestamp_diff"].std()
        feats["timestamp_diff_mean"] = trace["timestamp_diff"].mean()

        trace['duration'] = trace["end_time"] - trace["start_time"]
        trace["diff"] = trace["end_time"] - trace["start_time"]

        feats['diff_std'] = trace['diff'].std()
        feats['diff_mean'] = trace['diff'].mean()
        feats['diff_ptp'] = trace['diff'].max() - trace[
            'diff'].min()  # feats['diff'].dt.total_seconds().max() - feats['diff'].dt.total_seconds().min()
        #         feats['duration_min'] = trace['duration'].min().total_seconds()

        feats['duration_max'] = trace['diff'].max()  # .total_seconds()

        for i in ["host_ip", "service_name", "span_id"]:  # ,"endpoint_name","trace_id","parent_id"
            feats[f"{i}_max_id"] = \
            trace[i].value_counts().reset_index().sort_values([i, "count"], ascending=[False, True]).reset_index(
                drop=True)[:1][i].values[0]

            #             feats[f"{i}_max_id__"] = trace[i].value_counts().reset_index().sort_values(["count" , i], ascending=[False,True]).reset_index(drop=True)[:1][i].values[0]

            trace[f"{i}_max_id"] = \
            trace[i].value_counts().reset_index().sort_values([i, "count"], ascending=[False, True]).reset_index(
                drop=True)[:1][i].values[0]

            ids = trace[i].value_counts().reset_index().sort_values([i, "count"], ascending=[False, True]).reset_index(
                drop=True)[:1][i].values[0]
            feats[f"{i}_max_id_diff_max"] = trace[trace[i] == ids]['diff'].max()

        trace['time'] = trace.apply(lambda x: x['end_time'] - x['start_time'], axis=1)
        trace['endpoint'] = trace['endpoint_name'].apply(get_endpoint)

        ids = trace['host_ip'].value_counts().reset_index().sort_values(['host_ip', "count"],
                                                                        ascending=[False, True]).reset_index(drop=True)[
              :1]['host_ip'].values[0]
        trace_ = trace[trace['host_ip'] == ids]

        feats[f'trace_ip_max_id_cnt'] = len(trace_)

        for j in ['endpoint_name', 'trace_id', 'span_id', 'parent_id', 'endpoint']:
            feats[f"trace_{j}_nunique_ip_max_id"] = trace_[j].agg('nunique')
            feats[f"trace_{j}_entropy_ip_max_id"], feats[f"trace_{j}_max_ip_max_id"], feats[f"trace_{j}_min_ip_max_id"], \
            feats[f"trace_{j}_median_ip_max_id"] = calculate_entropy(trace_[j])

        for stats_func in ['mean', 'std', 'skew', 'kurt']:
            feats[f"trace_timestamp_{stats_func}_ip_max_id"] = trace_['timestamp'].apply(stats_func)

        trace_['diff'] = trace_['timestamp'].diff()
        trace_['start_diff'] = trace_['start_time'].diff()
        for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
            feats[f"trace_time_diff_{stats_func}_ip_max_id"] = trace_['diff'].apply(stats_func)
            feats[f"trace_time_{stats_func}_ip_max_id"] = trace_['time'].apply(stats_func)
            feats[f"trace_start_time_diff_{stats_func}_ip_max_id"] = trace_['start_diff'].apply(stats_func)

        feats['have_trace'] = 1
        #         feats['trace_length'] = len(trace)
        feats[f"trace_code_std"] = trace['status_code'].apply("std")

        feats['trace_lasting_time_ptp'] = trace['time'].apply('ptp')
        feats['trace_lasting_time_std'] = trace['time'].apply('std')
        feats['trace_lasting_time_mean'] = trace['time'].apply('mean')
        feats['trace_lasting_time_kurt'] = trace['time'].apply('kurt')
        feats['trace_lasting_time_skew'] = trace['time'].apply('skew')

        trace_code = trace[trace['status_code'] != 200]
        if len(trace_code) > 0:
            feats[f"trace_error_code_num"] = len(trace_code)
            df_temp = trace_code.groupby(['timestamp'])['time'].mean()
            feats['error_lasting_time_mean'] = df_temp.mean()
            feats['error_lasting_time_std'] = df_temp.std()
            feats['error_lasting_time_sum'] = trace_code['time'].sum()
            feats['error_lasting_time_ptp'] = trace_code['time'].max() - trace_code['time'].min()
            for service, df in list(trace_code.groupby('service_name')):
                if len(df) > 0:
                    feats[f'have_trace_code_{service}'] = 1
                else:
                    feats[f'have_trace_code_{service}'] = 0

                feats[f'error_code_cnt_{service}'] = len(df)
                feats[f'error_code_lasting_time_mean_{service}'] = df['time'].mean()
                feats[f'error_code_lasting_time_std_{service}'] = df['time'].std()
                feats[f'error_code_lasting_time_ptp_{service}'] = df['time'].max() - df['time'].min()
                feats[f'error_code_lasting_time_sum_{service}'] = df['time'].sum()

        tmp_list = list(trace['endpoint_name'].unique())

        if len(tmp_list) == 1 and 'Mysql/JDBI/Connection/close' in tmp_list:
            feats['endpoint_name_flag'] = 1
        else:
            feats['endpoint_name_flag'] = 0

        cnt = 0
        for i, error in enumerate(error_service):
            trace_error = trace[trace['service_name'] == error]
            if len(trace_error) > 0:
                cnt += 1
        feats[f'trace_error_service_cnt'] = cnt

        for stats_func in ['mean']:
            feats[f"trace_timestamp_{stats_func}"] = trace['timestamp'].apply(stats_func)

        for stats_func in ['nunique']:
            for i in ['host_ip', 'service_name', 'endpoint_name', 'trace_id', 'span_id', 'parent_id', 'start_time',
                      'end_time']:
                feats[f"trace_{i}_{stats_func}"] = trace[i].agg(stats_func)
                feats[f"trace_{i}_entropy"], feats[f"trace_{i}_max"], feats[f"trace_{i}_min"], feats[
                    f"trace_{i}_median"] = calculate_entropy(trace[i])

        for service, df in list(trace.groupby('service_name')):
            if len(df) > 0:
                feats[f'have_trace_{service}'] = 1
                feats[f'trace_service_cnt_{service}'] = len(df)

                for j in ['host_ip', 'endpoint_name', 'trace_id', 'span_id', 'parent_id', 'start_time', 'end_time']:
                    feats[f"trace_{j}_nunique_{service}"] = df[j].agg('nunique')
                    feats[f"trace_{j}_entropy_{service}"], feats[f"trace_{j}_max_{service}"], feats[
                        f"trace_{j}_min_{service}"], feats[f"trace_{j}_median_{service}"] = calculate_entropy(df[j])

                for stats_func in ['mean', 'std', 'skew', 'kurt']:
                    feats[f"trace_timestamp_{stats_func}_{service}"] = df['timestamp'].apply(stats_func)

                df['diff'] = df['timestamp'].diff()
                df['start_diff'] = df['start_time'].diff()
                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                    feats[f"trace_time_diff_{stats_func}_{service}"] = df['diff'].apply(stats_func)
                    feats[f"trace_time_{stats_func}_{service}"] = df['time'].apply(stats_func)
                    feats[f"trace_start_time_diff_{stats_func}_{service}"] = df['start_diff'].apply(stats_func)

                for j in range(1, 4):
                    trace_ip = df[df['host_ip'].str.split('.').str[2] == str(j)]
                    if len(trace_ip) > 0:
                        feats[f'trace_serviceip_cnt_{service}_{j}'] = len(trace_ip)
                        trace_ip['diff'] = trace_ip['timestamp'].diff()
                        trace_ip['start_diff'] = trace_ip['start_time'].diff()
                        for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                            feats[f"trace_time_diff_{stats_func}_{service}_{j}"] = trace_ip['diff'].apply(stats_func)
                            feats[f"trace_time_{stats_func}_{service}_{j}"] = trace_ip['time'].apply(stats_func)
                            feats[f"trace_start_time_diff_{stats_func}_{service}_{j}"] = trace_ip['start_diff'].apply(
                                stats_func)
                        for stats_func in ['nunique']:
                            for i in ['trace_id', 'span_id', 'parent_id', 'endpoint_name', 'endpoint']:
                                feats[f"trace_{service}_{j}_{stats_func}_{i}"] = trace_ip[i].agg(stats_func)

                        for end, df_ in list(trace_ip.groupby('endpoint')):
                            if len(df_) > 0:
                                feats[f'trace_end_cnt_{service}_{end}_{j}'] = len(df_)

                                df_['diff'] = df_['timestamp'].diff()
                                df_['start_diff'] = df_['start_time'].diff()
                                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                                    feats[f"trace_time_diff_{stats_func}_{service}_{end}_{j}"] = df_['diff'].apply(
                                        stats_func)
                                    feats[f"trace_time_{stats_func}_{service}_{end}_{j}"] = df_['time'].apply(
                                        stats_func)
                                    feats[f"trace_start_time_diff_{stats_func}_{service}_{end}_{j}"] = df_[
                                        'start_diff'].apply(stats_func)

                for end, df_ in list(df.groupby('endpoint')):
                    if len(df_) > 0:
                        feats[f'trace_end_cnt_{service}_{end}'] = len(df_)

                        df_['diff'] = df_['timestamp'].diff()
                        df_['start_diff'] = df_['start_time'].diff()
                        for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                            feats[f"trace_time_diff_{stats_func}_{service}_{end}"] = df_['diff'].apply(stats_func)
                            feats[f"trace_time_{stats_func}_{service}_{end}"] = df_['time'].apply(stats_func)
                            feats[f"trace_start_time_diff_{stats_func}_{service}_{end}"] = df_['start_diff'].apply(
                                stats_func)
                        for stats_func in ['nunique']:
                            for i in ['trace_id', 'span_id', 'parent_id', 'host_ip']:
                                feats[f"trace_{service}_{end}_{stats_func}_{i}"] = df_[i].agg(stats_func)

        for end, df in list(trace.groupby('endpoint')):
            if len(df) > 0:
                feats[f'trace_end_cnt_{end}'] = len(df)

                df['diff'] = df['timestamp'].diff()
                df['start_diff'] = df['start_time'].diff()
                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                    feats[f"trace_time_diff_{stats_func}_{end}"] = df['diff'].apply(stats_func)
                    feats[f"trace_time_{stats_func}_{end}"] = df['time'].apply(stats_func)
                    feats[f"trace_start_time_diff_{stats_func}_{end}"] = df['start_diff'].apply(stats_func)

                for stats_func in ['nunique']:
                    for i in ['service_name', 'trace_id', 'span_id', 'parent_id', 'host_ip', 'endpoint_name']:
                        feats[f"trace_{end}_{stats_func}_{i}"] = df[i].agg(stats_func)

        trace_end = trace[trace['endpoint_name'].isin(end_name)]
        for end, df in list(trace_end.groupby('endpoint_name')):
            if len(df) > 0:
                feats[f'trace_end_cnt_{end}'] = len(df)

                df['diff'] = df['timestamp'].diff()
                df['start_diff'] = df['start_time'].diff()
                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                    feats[f"trace_time_diff_{stats_func}_{end}"] = df['diff'].apply(stats_func)
                    feats[f"trace_time_{stats_func}_{end}"] = df['time'].apply(stats_func)
                    feats[f"trace_start_time_diff_{stats_func}_{end}"] = df['start_diff'].apply(stats_func)

                for stats_func in ['nunique']:
                    for i in ['service_name', 'trace_id', 'span_id', 'parent_id', 'host_ip']:
                        feats[f"trace_{end}_{stats_func}_{i}"] = df[i].agg(stats_func)

        for ip, df in list(trace.groupby('host_ip')):
            if len(df) > 0:
                feats[f'trace_ip_cnt_{ip}'] = len(df)

        for j in range(1, 4):
            trace_ip_ = trace[trace['host_ip'].str.split('.').str[2] == str(j)]
            if len(trace_ip_) > 0:
                feats[f'trace_host_ip_cnt_{j}'] = len(trace_ip_)

                trace_ip_['diff'] = trace_ip_['timestamp'].diff()
                trace_ip_['start_diff'] = trace_ip_['start_time'].diff()
                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                    feats[f"trace_time_diff_{stats_func}_{j}"] = trace_ip_['diff'].apply(stats_func)
                    feats[f"trace_time_{stats_func}_{j}"] = trace_ip_['time'].apply(stats_func)
                    feats[f"trace_start_time_diff_{stats_func}_{j}"] = trace_ip_['start_diff'].apply(stats_func)

                for stats_func in ['nunique']:
                    for i in ['service_name', 'trace_id', 'span_id', 'parent_id', 'endpoint_name', 'endpoint']:
                        feats[f"trace_{j}_{stats_func}_ip{i}"] = trace_ip_[i].agg(stats_func)



    else:
        feats['have_trace'] = 0
    #         feats['trace_length'] = -1

    return feats


all_ids = set([i.split("_")[0] for i in os.listdir(path+"metric/") ]) |\
          set([i.split("_")[0] for i in os.listdir(path+"log/")   ]) |\
          set([i.split("_")[0] for i in os.listdir(path+"trace/")  ])
all_ids = list(all_ids)

if ".ipynb" in all_ids:
    all_ids.remove(".ipynb")
    
print("IDs Length =", len(all_ids))

data = pd.DataFrame(Parallel(n_jobs=60, backend="multiprocessing")(delayed(processing_feature)(f) for f in tqdm(all_ids)))

data.to_parquet('../model/tmp_data/trace_feat.pqt')