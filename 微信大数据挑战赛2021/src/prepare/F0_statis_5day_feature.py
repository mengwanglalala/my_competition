import os
import pandas as pd
import gc
from tqdm import tqdm
BASE_PATH = '../../data/feather_data/'
OUT_PATH = "../../data/fea_data/"
chache_PATH = '../../data/cache/'
action_data = pd.read_feather(BASE_PATH + 'all_user_action_drop_duplicate_first.feather', columns=None, use_threads=True)
feed = pd.read_feather(BASE_PATH + 'feed_info.feather', columns=None, use_threads=True)
test = pd.read_feather(BASE_PATH + 'test_a.feather', columns=None, use_threads=True)

ACTION_LIST =["read_comment", "like", "click_avatar","favorite","forward", "comment", "follow"]

print("==> Loading training data")
# action_data = pd.read_csv(BASE_PATH + 'wedata/wechat_algo_data1/user_action.csv')
# # action_data = pd.read_csv(BASE_PATH + 'down_sampling_data/sampling_data.csv')
# feed = pd.read_csv(BASE_PATH + 'wedata/wechat_algo_data1/feed_info.csv')
# test_data_a = pd.read_csv(BASE_PATH + 'wedata/wechat_algo_data1/test_a.csv')
# test_data_b = pd.read_csv(BASE_PATH + 'wedata/wechat_algo_data1/test_b.csv')

# 合并数据集,统一处理
test_data = test
test_data[ACTION_LIST] = 0
test_data['date_'] = 15

data = pd.concat((action_data, test_data)).reset_index(drop=True)
print('==>initial_data_shape:', action_data.shape, test_data.shape, data.shape)

feed[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
    feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
feed['bgm_song_id'] = feed['bgm_song_id'].astype('int64')
feed['bgm_singer_id'] = feed['bgm_singer_id'].astype('int64')


data = data.merge(
    feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']],
    how='left',
    on='feedid')


## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
## 视频时长是秒，转换成毫秒，才能与play、stay做运算
df = data
df['videoplayseconds'] *= 1000
del data
del test_data
del action_data
del feed
gc.collect()
## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
df['is_finish'] = (df['play'] >= 0.8*df['videoplayseconds']).astype('int8')
df['play_times'] = df['play'] / df['videoplayseconds']
play_cols = [
    'is_finish', 'play_times', 'play', 'stay'
]
n_day = 5
max_day = 15

n_day = 5

# for stat_cols in tqdm([
#     ['userid'],
#     ['feedid'],
#     ['authorid'],
#     ['bgm_song_id'],
#     ['bgm_singer_id'],
#     ['userid', 'authorid'],
#     ['userid', 'bgm_song_id'],
#     ['userid', 'bgm_singer_id'],
# ]):
#     f = '_'.join(stat_cols)
#     stat_df = pd.DataFrame()
#     for target_day in range(2, max_day + 1):
#         left, right = max(target_day - n_day, 1), target_day - 1
#         tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
#         tmp['date_'] = target_day
#         tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')
#         g = tmp.groupby(stat_cols)
#         tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')
#         feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]
#         for x in play_cols[1:]:
#             for stat in ['max', 'mean']:
#                 tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
#                 feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))
#         for y in y_list[:4]:
#             tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')
#             tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')
#             feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])
#         tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
#         stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
#         del g, tmp
#     df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
#     del stat_df


#自己加入, 'bgm_song_id', 'bgm_singer_id'和几个多出的组合

#for f in tqdm(['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']):
for f in tqdm(['userid', 'feedid', 'authorid']):
    df[f + '_count'] = df[f].map(df[f].value_counts())

for f1, f2 in tqdm([

    ['userid', 'feedid'],
    #['userid', 'bgm_song_id'],
    #['userid', 'bgm_singer_id'],
    ['userid', 'authorid'],

]):

    df['{}_in_{}_nunique'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')
    df['{}_in_{}_nunique'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')
    #自己加的
#     df['{}_in_{}std'.format(f1, f2)] = df.groupby(f2)[f1].transform('std')
#     df['{}_in_{}std'.format(f2, f1)] = df.groupby(f1)[f2].transform('std')
#     df['{}_in_{}max'.format(f1, f2)] = df.groupby(f2)[f1].transform('max')
#     df['{}_in_{}max'.format(f2, f1)] = df.groupby(f1)[f2].transform('max')
#     df['{}_in_{}min'.format(f1, f2)] = df.groupby(f2)[f1].transform('min')
#     df['{}_in_{}min'.format(f2, f1)] = df.groupby(f1)[f2].transform('min')
    #后加
    # df['{}_in_{}min'.format(f1, f2)] = df.groupby(f2)[f1].transform('mean')
    # df['{}_in_{}min'.format(f2, f1)] = df.groupby(f1)[f2].transform('mean')
    # df['{}_in_{}min'.format(f1, f2)] = df.groupby(f2)[f1].transform('count')
    # df['{}_in_{}min'.format(f2, f1)] = df.groupby(f1)[f2].transform('count')

for f1, f2 in tqdm([
    ['userid', 'authorid'],
    #自己加的
#     ['userid', 'bgm_song_id'],
#     ['userid', 'bgm_singer_id'],

]):

    df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['date_'].transform('count')
    df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)
    df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)

df['videoplayseconds_in_userid_mean'] = df.groupby('userid')['videoplayseconds'].transform('mean')
df['videoplayseconds_in_authorid_mean'] = df.groupby('authorid')['videoplayseconds'].transform('mean')
df['feedid_in_authorid_nunique'] = df.groupby('authorid')['feedid'].transform('nunique')


## 内存够用的不需要做这一步
# df = reduce_mem(df, [f for f in df.columns if f not in ['date_'] + play_cols + y_list])

# 不加统计特征似乎与lgb的融合效果更好？？？待测试
# cols = ['userid_5day_count', 'userid_5day_finish_rate', 'userid_5day_play_times_max', 'userid_5day_play_times_mean',
#         'userid_5day_play_max', 'userid_5day_play_mean', 'userid_5day_stay_max', 'userid_5day_stay_mean',
#         'userid_5day_read_comment_sum', 'userid_5day_read_comment_mean', 'userid_5day_like_sum',
#         'userid_5day_like_mean', 'userid_5day_click_avatar_sum', 'userid_5day_click_avatar_mean',
#         'userid_5day_forward_sum', 'userid_5day_forward_mean', 'feedid_5day_count', 'feedid_5day_finish_rate',
#         'feedid_5day_play_times_max', 'feedid_5day_play_times_mean', 'feedid_5day_play_max', 'feedid_5day_play_mean',
#         'feedid_5day_stay_max', 'feedid_5day_stay_mean', 'feedid_5day_read_comment_sum', 'feedid_5day_read_comment_mean',
#         'feedid_5day_like_sum', 'feedid_5day_like_mean', 'feedid_5day_click_avatar_sum', 'feedid_5day_click_avatar_mean',
#         'feedid_5day_forward_sum', 'feedid_5day_forward_mean', 'authorid_5day_count', 'authorid_5day_finish_rate',
#         'authorid_5day_play_times_max', 'authorid_5day_play_times_mean', 'authorid_5day_play_max', 'authorid_5day_play_mean',
#         'authorid_5day_stay_max', 'authorid_5day_stay_mean', 'authorid_5day_read_comment_sum', 'authorid_5day_read_comment_mean',
#         'authorid_5day_like_sum', 'authorid_5day_like_mean', 'authorid_5day_click_avatar_sum', 'authorid_5day_click_avatar_mean',
#         'authorid_5day_forward_sum', 'authorid_5day_forward_mean', 'userid_authorid_5day_count', 'userid_authorid_5day_finish_rate',
#         'userid_authorid_5day_play_times_max', 'userid_authorid_5day_play_times_mean', 'userid_authorid_5day_play_max',
#         'userid_authorid_5day_play_mean', 'userid_authorid_5day_stay_max', 'userid_authorid_5day_stay_mean', 'userid_authorid_5day_read_comment_sum',
#         'userid_authorid_5day_read_comment_mean', 'userid_authorid_5day_like_sum', 'userid_authorid_5day_like_mean',
#         'userid_authorid_5day_click_avatar_sum', 'userid_authorid_5day_click_avatar_mean', 'userid_authorid_5day_forward_sum',
#         'userid_authorid_5day_forward_mean', 'userid_count', 'feedid_count', 'authorid_count', 'userid_in_feedid_nunique',
#         'feedid_in_userid_nunique', 'userid_in_authorid_nunique', 'authorid_in_userid_nunique', 'userid_authorid_count',
#         'userid_in_authorid_count_prop', 'authorid_in_userid_count_prop', 'videoplayseconds_in_userid_mean',
#         'videoplayseconds_in_authorid_mean', 'feedid_in_authorid_nunique']

cols = ['userid_in_feedid_nunique',
            'feedid_in_userid_nunique', 'userid_in_authorid_nunique', 'authorid_in_userid_nunique',
            'userid_authorid_count',
            'userid_in_authorid_count_prop', 'authorid_in_userid_count_prop', 'videoplayseconds_in_userid_mean',
            'videoplayseconds_in_authorid_mean', 'feedid_in_authorid_nunique']
cols += ['feedid','userid','date_']
print('extract finished')

df_out = df[cols]
del df
gc.collect()
print('finished')

print('ready to save')
#df[cols].to_csv("../../feature_data/statis_5_day_featrue_a_and_b.csv",index=False)
compressed_file = OUT_PATH + "v0_statis_5_day_featrue_all_data_drop_duplicate_first.feather"
df_out.to_feather(compressed_file,compression ='zstd',compression_level =2)
print('finished')