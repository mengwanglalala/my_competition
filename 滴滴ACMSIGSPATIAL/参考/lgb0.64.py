import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from lightgbm.sklearn import LGBMClassifier
from collections import defaultdict
import gc
import time
from sklearn.decomposition import PCA
pd.set_option('display.max_columns', None)

'''
[0.6188753802455995, 0.6175623324902096, 0.710624311564186, 0.6871882972451463]
0.6436625438826545
'''

def reduce_mem(df, cols):

    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in tqdm(cols):

        col_type = df[col].dtypes

        if col_type != object:

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

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    gc.collect()

    return df

## 从官方baseline里面抽出来的评测函数
def uAUC(labels, preds, user_id_list):

    """Calculate user AUC"""

    user_pred = defaultdict(lambda: [])

    user_truth = defaultdict(lambda: [])

    for idx, truth in enumerate(labels):

        user_id = user_id_list[idx]

        pred = preds[idx]

        truth = labels[idx]

        user_pred[user_id].append(pred)

        user_truth[user_id].append(truth)



    user_flag = defaultdict(lambda: False)

    for user_id in set(user_id_list):

        truths = user_truth[user_id]

        flag = False

        # 若全是正样本或全是负样本，则flag为False

        for i in range(len(truths) - 1):

            if truths[i] != truths[i + 1]:

                flag = True

                break

        user_flag[user_id] = flag



    total_auc = 0.0

    size = 0.0

    for user_id in user_flag:

        if user_flag[user_id]:

            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))

            total_auc += auc

            size += 1.0

    user_auc = float(total_auc)/size

    return user_auc

y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15

## 读取训练集
train = pd.read_csv('../src/data/wechat_algo_data1/user_action.csv')
print(train.shape)
for y in y_list:
    print(y, train[y].mean())



## 读取测试集
test = pd.read_csv('../src/data/wechat_algo_data1/test_a.csv')
test['date_'] = max_day
print(test.shape)

## 合并处理
df = pd.concat([train, test], axis=0, ignore_index=True)
print(df.head(3))



## 读取视频信息表
feed_info = pd.read_csv('../src/data/wechat_algo_data1/feed_info.csv')
## 此份baseline只保留这三列
feed_info = feed_info[[
    'feedid', 'authorid', 'videoplayseconds','bgm_song_id', 'bgm_singer_id'
]]

# w2v_feature = []
# # 512 维度特征
# feed_emb = pd.read_csv('../src/data/wechat_algo_data1/feed_embeddings_process.csv')
# #PCA处理前需要去掉feedid
# df1 = feed_emb.drop(columns=['feedid'])
# pca = PCA(n_components=32)  # 降维为32维
# reduce_x = pca.fit_transform(np.array(df1))
# df_emb = pd.DataFrame(reduce_x)
# headers = df_emb.columns.to_list()
# for col in headers:
#     df_emb.rename(columns={col: 'feedemb' + str(col)}, inplace=True)
# #df.columns = df.columns.map(str)
# df_emb['feedid'] = feed_emb['feedid']
# feed_info = df_emb.set_index('feedid')
# embed_featrure = [column for column in feed_info]
# df = df.join(feed_info, on="feedid", how="left")  # 特征未进行处理

#统计特征
# stats = pd.read_csv('../src/data/statis_feature/userid_7day_feature.csv')
# df = df.merge(stats, on=['userid','date_'], how='left')

df = df.merge(feed_info, on='feedid', how='left')
## 视频时长是秒，转换成毫秒，才能与play、stay做运算
df['videoplayseconds'] *= 1000

## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int8')
df['play_times'] = df['play'] / df['videoplayseconds']
play_cols = [
    'is_finish', 'play_times', 'play', 'stay'
]



train = df[~df['read_comment'].isna()].reset_index(drop=True)

test = df[df['read_comment'].isna()].reset_index(drop=True)



## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
n_day = 5
for stat_cols in tqdm([
    ['userid'],
    ['feedid'],
    ['authorid'],
    ['userid', 'authorid']
]):
    f = '_'.join(stat_cols)
    stat_df = pd.DataFrame()
    for target_day in range(2, max_day + 1):
        left, right = max(target_day - n_day, 1), target_day - 1
        tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
        tmp['date_'] = target_day
        tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')
        g = tmp.groupby(stat_cols)
        tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')
        feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]

        for x in play_cols[1:]:

            for stat in ['max', 'mean']:

                tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)

                feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))



        for y in y_list[:4]:

            tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')

            tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')

            feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])



        tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)

        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)

        del g, tmp
    df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
    del stat_df
    gc.collect()





## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行

for f in tqdm(['userid', 'feedid', 'authorid']):

    df[f + '_count'] = df[f].map(df[f].value_counts())

for f1, f2 in tqdm([

    ['userid', 'feedid'],

    ['userid', 'authorid']

]):

    df['{}_in_{}_nunique'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')

    df['{}_in_{}_nunique'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')

for f1, f2 in tqdm([

    ['userid', 'authorid']

]):

    df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['date_'].transform('count')

    df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)

    df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)

df['videoplayseconds_in_userid_mean'] = df.groupby('userid')['videoplayseconds'].transform('mean')

df['videoplayseconds_in_authorid_mean'] = df.groupby('authorid')['videoplayseconds'].transform('mean')

df['feedid_in_authorid_nunique'] = df.groupby('authorid')['feedid'].transform('nunique')



## 内存够用的不需要做这一步

#df = reduce_mem(df, [f for f in df.columns if f not in ['date_'] + play_cols + y_list])


cols = [f for f in df.columns if f not in ['date_'] + play_cols + y_list]

print(train[cols].shape)

trn_x = train[train['date_'] < 14].reset_index(drop=True)

val_x = train[train['date_'] == 14].reset_index(drop=True)

##################### 线下验证 #####################

uauc_list = []

r_list = []

for y in y_list[:4]:

    print('=========', y, '=========')

    t = time.time()

    clf = LGBMClassifier(

        learning_rate=0.05,

        n_estimators=5000,

        num_leaves=63,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=2021,

        metric='None'

    )

    clf.fit(

        trn_x[cols], trn_x[y],

        eval_set=[(val_x[cols], val_x[y])],

        eval_metric='auc',

        early_stopping_rounds=100,

        verbose=50

    )

    val_x[y + '_score'] = clf.predict_proba(val_x[cols])[:, 1]

    val_uauc = uAUC(val_x[y], val_x[y + '_score'], val_x['userid'])

    uauc_list.append(val_uauc)

    print(val_uauc)

    r_list.append(clf.best_iteration_)

    print('runtime: {}\n'.format(time.time() - t))



weighted_uauc = 0.4 * uauc_list[0] + 0.3 * uauc_list[1] + 0.2 * uauc_list[2] + 0.1 * uauc_list[3]

print(uauc_list)

print(weighted_uauc)



##################### 全量训练 #####################

r_dict = dict(zip(y_list[:4], r_list))

for y in y_list[:4]:

    print('=========', y, '=========')

    t = time.time()

    clf = LGBMClassifier(

        learning_rate=0.05,

        n_estimators=r_dict[y],

        num_leaves=63,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=2021

    )

    clf.fit(

        train[cols], train[y],

        eval_set=[(train[cols], train[y])],

        early_stopping_rounds=r_dict[y],

        verbose=100

    )

    test[y] = clf.predict_proba(test[cols])[:, 1]

    print('runtime: {}\n'.format(time.time() - t))

test[['userid', 'feedid'] + y_list[:4]].to_csv(
    'sub_%.6f_%.6f_%.6f_%.6f_%.6f.csv' % (weighted_uauc, uauc_list[0], uauc_list[1], uauc_list[2], uauc_list[3]),
    index=False
)