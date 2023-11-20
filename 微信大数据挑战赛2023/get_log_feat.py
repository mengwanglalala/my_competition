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


##baseline  19个
def processing_feature(file):
    log = pd.DataFrame()
    if os.path.exists(path + f"log/{file}_log.csv"):
        log = pd.read_csv(path + f"log/{file}_log.csv").sort_values(by=['timestamp']).reset_index(drop=True)

    feats = {"id": file}

    if len(log) > 0:
        feats['log_length'] = len(log)
        log['service_count'] = log["service"].count()
        log['time_diff'] = log['timestamp'].shift(-1) - log['timestamp']
        log.loc[log.index[-1], 'time_diff'] = 0
        grouped = log.groupby("service")["time_diff"]
        feats['log_mean_time_diff'] = grouped.mean().to_numpy()[0]
        feats['log_std_time_diff'] = grouped.std().to_numpy()[0]
        feats['log_min_time_diff'] = grouped.min().to_numpy()[0]
        feats['log_max_time_diff'] = grouped.max().to_numpy()[0]
        feats['service_max_id'] = log["service"].value_counts().reset_index().sort_values(["service", "count"],
                                                                                          ascending=[False,
                                                                                                     True]).reset_index(
            drop=True)[:1]["service"].values[0]  # , ascending=[False, True]

        log['message_length'] = log['message'].fillna("").map(len)
        feats['message_length_mean'] = log['message_length'].mean()
        feats[f'message_length_max'] = log['message_length'].max()
        feats[f'message_length_min'] = log['message_length'].min()
        feats[f'message_length_std'] = log['message_length'].std()
        feats[f'log_service_cnt'] = log['service'].agg('nunique')

        cnt = 0
        for i, error in enumerate(error_service):
            log_error = log[log['service'] == error]
            if len(log_error) > 0:
                cnt += 1
        feats[f'log_error_service_cnt_'] = cnt

        for service, df in list(log.groupby('service')):
            if len(df) > 0:
                feats[f'log_service_cnt_{service}'] = len(df)

                df['diff'] = df['timestamp'].diff()
                for stats_func in ['mean', 'std', 'skew', 'kurt', 'ptp']:
                    feats[f"log_time_diff_{stats_func}_{service}"] = df['diff'].apply(stats_func)

                feats[f'message_length_mean{service}'] = df['message_length'].mean()
                feats[f'message_length_max{service}'] = df['message_length'].max()
                feats[f'message_length_min{service}'] = df['message_length'].min()
                feats[f'message_length_std{service}'] = df['message_length'].std()

    else:
        feats['log_length'] = -1

    return feats

all_ids = set([i.split("_")[0] for i in os.listdir(path+"metric/") ]) |\
          set([i.split("_")[0] for i in os.listdir(path+"log/")   ]) |\
          set([i.split("_")[0] for i in os.listdir(path+"trace/")  ])
all_ids = list(all_ids)

if ".ipynb" in all_ids:
    all_ids.remove(".ipynb")
    
print("IDs Length =", len(all_ids))

data = pd.DataFrame(Parallel(n_jobs=60, backend="multiprocessing")(delayed(processing_feature)(f) for f in tqdm(all_ids)))

data.to_parquet('../model/tmp_data/log_feat.pqt')