import os
import pandas as pd
from tqdm import tqdm
#wbdc2021/data/wedata/wechat_algo_data1
BASE_PATH = '../../../wbdc2021/data/wedata/'
#读入初赛数据
action_data = pd.read_csv(BASE_PATH + 'wechat_algo_data1/user_action.csv')
feed = pd.read_csv(BASE_PATH + 'wechat_algo_data1/feed_info.csv')
test_data_a = pd.read_csv(BASE_PATH + 'wechat_algo_data1/test_a.csv')
test_data_b = pd.read_csv(BASE_PATH + 'wechat_algo_data1/test_b.csv')
action_data.head()

#读入复赛数据
action_data2 = pd.read_csv(BASE_PATH + 'wechat_algo_data2/user_action.csv')
feed2 = pd.read_csv(BASE_PATH + 'wechat_algo_data2/feed_info.csv')
test_data_a2 = pd.read_csv(BASE_PATH + 'wechat_algo_data2/test_a.csv')

OUT_PATH = "../../data/feather_data/"
action_data2.to_feather(OUT_PATH+"user_action_just_fusai.feather",compression ='zstd',compression_level =2)
print('save finished')

all_action_data = pd.concat((action_data2, action_data)).reset_index(drop=True)
OUT_PATH = "../../data/feather_data/"
all_action_data.to_feather(OUT_PATH+"all_user_action.feather",compression ='zstd',compression_level =2)
print('save finished')

feed2.to_feather(OUT_PATH+"feed_info.feather",compression ='zstd',compression_level =2)
print('save finished')

test_pre = pd.concat((test_data_a, test_data_b)).reset_index(drop=True)
all_test_data = pd.concat((test_data_a2, test_pre)).reset_index(drop=True)
all_test_data.to_feather(OUT_PATH+"all_test_data.feather",compression ='zstd',compression_level =2)
print('save finished')

test_data_a2.to_feather(OUT_PATH+"test_a.feather",compression ='zstd',compression_level =2)
print('save finished')

action_data2.to_feather(OUT_PATH+"user_action.feather",compression ='zstd',compression_level =2)
print('save finished')

feed_embedding = pd.read_csv(BASE_PATH + 'wechat_algo_data2/feed_embeddings.csv')
feed_embedding.to_feather(OUT_PATH+"feed_embeddings.feather",compression ='zstd',compression_level =2)
print('save finished')