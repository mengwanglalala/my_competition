import tensorflow as tf

print(tf.test.gpu_device_name())
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import gc
from time import time
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.python.keras.utils import multi_gpu_model

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()
print(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
from evaluation import evaluate_uAUC
# sys.path.append(os.path.join(BASE_DIR, '../model'))
# from model.mmoe import MMOE
from model.model import MMOE
from tensorflow.python.keras.initializers import RandomNormal, Zeros, TruncatedNormal, RandomUniform
from tqdm import tqdm
from tensorflow.python.keras.optimizers import Adam, Adagrad

print(tf.test.gpu_device_name())
print(tf.__version__)
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers

'''
线上：0.704221
'''
# GPU相关设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# 设置GPU按需增长
config = tf.ConfigProto()
# config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# sess = tf.compat.v1.Session(config=config)
# 用于构造特征的字段列表
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
ACTION_LIST = ["read_comment", "like", "click_avatar", "favorite", "forward", "comment", "follow"]
target = ["read_comment", "like", "click_avatar", "favorite", "forward", "comment", "follow"]
Tasks = ['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary']
BASE_PATH = '../../data/feather_data/'
FEA_PATH = "../../data/fea_data/"
FEA_PATH2 = "../../data/deepwalk_feature/"
chache_PATH = '../../data/cache/'
MODEL_PATH = '../../data/model/'
TRAN_PATH = '../../data/train_data/'
key2index = {}

target = ["read_comment", "like", "click_avatar", "favorite", "forward", "comment", "follow"]
Tasks = ['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary']
NUM_TASK = 7


# adam warmup策略
class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):
    """Warmup learning rate scheduler
    """

    def __init__(self, warmup_batches, init_lr, verbose=0):
        """Constructor for warmup learning rate scheduler

        Arguments:
            warmup_batches {int} -- Number of batch for warmup.
            init_lr {float} -- Learning rate after warmup.

        Keyword Arguments:
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count * self.init_lr / self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))


class AdamWarmup(optimizers.Optimizer):
    def __init__(self, decay_steps, warmup_steps, min_lr=0.0,
                 lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, kernel_weight_decay=0., bias_weight_decay=0.,
                 amsgrad=False, **kwargs):
        super(AdamWarmup, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.decay_steps = K.variable(decay_steps, name='decay_steps')
            self.warmup_steps = K.variable(warmup_steps, name='warmup_steps')
            self.min_lr = K.variable(min_lr, name='min_lr')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.kernel_weight_decay = K.variable(kernel_weight_decay, name='kernel_weight_decay')
            self.bias_weight_decay = K.variable(bias_weight_decay, name='bias_weight_decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_kernel_weight_decay = kernel_weight_decay
        self.initial_bias_weight_decay = bias_weight_decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        lr = K.switch(
            t <= self.warmup_steps,
            self.lr * (t / self.warmup_steps),
            self.lr * (1.0 - K.minimum(t, self.decay_steps) / self.decay_steps),
        )

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = m_t / (K.sqrt(v_t) + self.epsilon)

            if 'bias' in p.name or 'Norm' in p.name:
                if self.initial_bias_weight_decay > 0.0:
                    p_t += self.bias_weight_decay * p
            else:
                if self.initial_kernel_weight_decay > 0.0:
                    p_t += self.kernel_weight_decay * p
            p_t = p - lr_t * p_t

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates


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


sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'device']
dense_features = ['videoplayseconds']
w2v_features = []
tf_idf_feature = []
test_fea = []
keyword_features = []
##################读取id特征 ##############################

print("==> Loading training data")
t0 = time()

action_data = pd.read_feather(BASE_PATH + 'all_user_action_drop_duplicate_first.feather', columns=None,
                              use_threads=True)  # 有效
# feed = pd.read_feather(BASE_PATH + 'feed_info.feather', columns=None, use_threads=True)
# test_data = pd.read_feather(BASE_PATH + 'test_a.feather', columns=None, use_threads=True)
# 原始数据
BASE_PATH = '../../../wbdc2021/data/wedata/'
# action_data = pd.read_csv(BASE_PATH + 'wechat_algo_data2/user_action.csv')
feed = pd.read_csv(BASE_PATH + 'wechat_algo_data2/feed_info.csv')
test_data = pd.read_csv(BASE_PATH + 'wechat_algo_data2/test_a.csv')

t1 = time()
print('read_data_time:', t1 - t0)

# 合并数据集,统一处理
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
# 放在这里的原因是后面有labelencode操作,会导致id发生变化
submit = test_data[['userid', 'feedid']]
del feed
del action_data
del test_data
gc.collect()

data = reduce_mem(data, [f for f in data.columns if f not in ['userid', 'feedid', 'date_'] + target])
print('==> data base size:', data.info(memory_usage='deep'))
print('==>data_base_fea_shape:', data.shape)

t2 = time()
print('read_base_data_cost_time:', t2 - t1)

#################################################merge特征（需要优化内存）####################################################################
# v6 加入w2v特征 复赛线上还掉了1k，用单纯只预测前四个target可能有涨
# n2v_feature = pd.read_pickle(FEA_PATH + 'w2v_tag_.pkl')
# data = data.merge(n2v_feature, on=["feedid"], how="left")
# w2v_features +=  ['w2v_tag_0','w2v_tag_1','w2v_tag_2','w2v_tag_3','w2v_tag_4','w2v_tag_5','w2v_tag_6','w2v_tag_7'] # 初赛线下涨了4-5k
# del n2v_feature
# gc.collect()
# data = reduce_mem(data, w2v_features)

# v0 统计特征
print('add statis feature')
statis_fea = pd.read_feather(FEA_PATH + 'v0_statis_5_day_featrue_all_data_drop_duplicate_first.feather', columns=None,
                             use_threads=True)  # 10
print('==> statis feature size:', statis_fea.info(memory_usage='deep'))
cols = ['userid_in_feedid_nunique',
        'feedid_in_userid_nunique', 'userid_in_authorid_nunique', 'authorid_in_userid_nunique',
        'userid_authorid_count',
        'userid_in_authorid_count_prop', 'authorid_in_userid_count_prop', 'videoplayseconds_in_userid_mean',
        'videoplayseconds_in_authorid_mean', 'feedid_in_authorid_nunique']
statis_fea = reduce_mem(statis_fea, cols)
print('statis_fea: {:.2f} GB'.format(statis_fea.memory_usage().sum() / (1024 ** 3)))
data = data.merge(statis_fea, on=["userid", "feedid", "date_"], how="left")
dense_features += cols
del statis_fea
gc.collect()
print('==> data now size:', data.info(memory_usage='deep'))

# v1 512维度embedding
print('add statis 512embedding')
embedding_512_feature = pd.read_feather(FEA_PATH + 'v1_feed_embeddings_process.feather', columns=None, use_threads=True)
data = data.merge(embedding_512_feature, on=["feedid"], how="left")
embedding_512_feature = embedding_512_feature.set_index(["feedid"])
tf_idf_feature += embedding_512_feature.columns.to_list()  # [0:6]
del embedding_512_feature
gc.collect()
print(data[tf_idf_feature])
data = reduce_mem(data, tf_idf_feature)

# # v2 user history
# print('add user history')
# action_history = pd.read_feather(FEA_PATH + 'v2_5_day_feed_histoy.feather', columns=None, use_threads=True)
# print('==> data action_history size:', action_history.info(memory_usage='deep'))
# print('action_history: {:.2f} GB'.format(action_history.memory_usage().sum()/ (1024**3)))
# add_fea = ['user_history0', 'user_history1', 'user_history2', 'user_history3', 'user_history4', 'user_history5',
#            'user_history6', 'user_history7', 'user_history8', 'user_history9', 'user_history10', 'user_history11',
#            'user_history12', 'user_history13', 'user_history14', 'user_history15', 'user_history16',
#            'user_history17',
#            'user_history18', 'user_history19', 'user_history20', 'user_history21', 'user_history22',
#            'user_history23']
# action_history = reduce_mem(action_history,  action_history.columns.to_list())
# data = data.merge(action_history[add_fea + ["userid", "date_"]], on=["userid", "date_"], how="left")
# print('==> data now size:', data.info(memory_usage='deep'))

# data[add_fea] = data[add_fea].fillna(0, )
# data[add_fea] = data[add_fea].astype('int32')
# keyword_features += add_fea
# del action_history
# gc.collect()
# print(data[add_fea])
# data = reduce_mem(data, add_fea)

# v3 加入关键词特征,关键词特征已经labelencode过了
# keyword =  pd.read_feather(FEA_PATH + 'v3_1_key_word.feather', columns=None, use_threads=True) #pd.read_csv(BASE_PATH + 'features/key_word.csv')  # 18
# data = data.merge(keyword, on=["feedid"], how="left")
# keyword = keyword.set_index(["feedid"])
# keyword_features = keyword.columns.to_list()  # [0:10]
# del keyword

# tag =  pd.read_feather(FEA_PATH + 'v3_0_tag_list.feather', columns=None, use_threads=True) #pd.read_csv(BASE_PATH + 'features/tag_list.csv')  # 10
# data = data.merge(tag, on=["feedid"], how="left")
# tag = tag.set_index(["feedid"])
# keyword_features += tag.columns.to_list()  # [0:6]
# del tag
# gc.collect()
# print('add tag and key feature',data.shape)
# print(data[keyword_features])
# data = reduce_mem(data, keyword_features)

# v4点击率设置
# read_comment = pd.read_feather(FEA_PATH + 'v4_click_feature_just_fusai.feather', columns=None, use_threads=True)
# print(read_comment.head())
# data = data.merge(read_comment, on=["userid", "date_"], how="left")
# w2v_features += ['userid_read_comment_click', 'userid_like_click', 'userid_click_avatar_click', 'userid_forward_click']
# del read_comment
# gc.collect()
# print('add click rate feature',data.shape)
# data = reduce_mem(data, ['userid_read_comment_click', 'userid_like_click', 'userid_click_avatar_click', 'userid_forward_click'])

# v5 加入n2v特征
print('==>add n2v feature')
n2v_feature = pd.read_pickle(FEA_PATH + 'v5_1_userid_authorid_n2v.pkl')
n2v_feature = reduce_mem(n2v_feature, n2v_feature.columns.to_list())
print('==> n2v size:', n2v_feature.info(memory_usage='deep'))
print('n2v_feature: {:.2f} GB'.format(n2v_feature.memory_usage().sum() / (1024 ** 3)))
print('==>merge n2v feature')
data = data.merge(n2v_feature, on=["userid"], how="left")
n2v_feature = n2v_feature.set_index(["userid"])
tf_idf_feature += n2v_feature.columns.to_list()  # [0:6]
del n2v_feature
gc.collect()
print('data shape：', data.shape)
print('==> data now size:', data.info(memory_usage='deep'))

# v5 加入deepwalk特征
n2v_feature = pd.read_feather(FEA_PATH + 'fea_user_deepwalk.feather', columns=None, use_threads=True)  # 原始复赛数据
n2v_feature = reduce_mem(n2v_feature, n2v_feature.columns.to_list())
print('==> deepwalk size:', n2v_feature.info(memory_usage='deep'))
print('n2v_feature: {:.2f} GB'.format(n2v_feature.memory_usage().sum() / (1024 ** 3)))
print('==>add n2v feature')
data = data.merge(n2v_feature, on=["feedid"], how="left")
print('==>merge n2v feature finished')
n2v_feature = n2v_feature.set_index(["feedid"])
tf_idf_feature += n2v_feature.columns.to_list()  # [0:6]
print('==> n2v feature finished')
del n2v_feature
gc.collect()
print('==> data now size:', data.info(memory_usage='deep'))
data = reduce_mem(data, tf_idf_feature)
print('==> data now size:', data.info(memory_usage='deep'))

# print('==>merge n2v feature finished')
# n2v_feature = pd.read_feather(FEA_PATH2 + 'userid_authorid_userid_all_no_fusai_test_deepwalk_32.feather', columns=None, use_threads=True) #与n2v重复了
# n2v_feature = reduce_mem(n2v_feature,  n2v_feature.columns.to_list())
# print('==> deepwalk size:', n2v_feature.info(memory_usage='deep'))
# print('n2v_feature: {:.2f} GB'.format(n2v_feature.memory_usage().sum()/ (1024**3)))
# print('==>add n2v feature')
# data = data.merge(n2v_feature, on=["userid"], how="left")
# print('==>merge n2v feature finished')
# n2v_feature = n2v_feature.set_index(["userid"])
# tf_idf_feature += n2v_feature.columns.to_list()  # [0:6]
# print('==> n2v feature finished')
# del n2v_feature
# gc.collect()
# print('==> data now size:', data.info(memory_usage='deep'))
# data = reduce_mem(data, tf_idf_feature)
# print('==> data now size:', data.info(memory_usage='deep'))

# print('==>merge n2v feature finished')
# n2v_feature = pd.read_feather(FEA_PATH2 + 'userid_feedid_userid_all_no_fusai_test_deepwalk_32.feather', columns=None, use_threads=True) #feeduser 反过来
# n2v_feature = reduce_mem(n2v_feature,  n2v_feature.columns.to_list())
# print('==> deepwalk size:', n2v_feature.info(memory_usage='deep'))
# print('n2v_feature: {:.2f} GB'.format(n2v_feature.memory_usage().sum()/ (1024**3)))
# print('==>add n2v feature')
# data = data.merge(n2v_feature, on=["userid"], how="left")
# print('==>merge n2v feature finished')
# n2v_feature = n2v_feature.set_index(["userid"])
# tf_idf_feature += n2v_feature.columns.to_list()  # [0:6]
# print('==> n2v feature finished')
# del n2v_feature
# gc.collect()
# print('==> data now size:', data.info(memory_usage='deep'))
# data = reduce_mem(data, tf_idf_feature)
# print('==> data now size:', data.info(memory_usage='deep'))


# print('==>merge n2v feature finished')
# n2v_feature = pd.read_feather(FEA_PATH2 + 'feedid_userid_feedid_all_no_fusai_test_deepwalk_32.feather', columns=None, use_threads=True) #原始复赛数据
# n2v_feature = reduce_mem(n2v_feature,  n2v_feature.columns.to_list())
# print('==> deepwalk size:', n2v_feature.info(memory_usage='deep'))
# print('n2v_feature: {:.2f} GB'.format(n2v_feature.memory_usage().sum()/ (1024**3)))
# print('==>add n2v feature')
# data = data.merge(n2v_feature, on=["feedid"], how="left")
# print('==>merge n2v feature finished')
# n2v_feature = n2v_feature.set_index(["feedid"])
# tf_idf_feature += n2v_feature.columns.to_list()  # [0:6]
# print('==> n2v feature finished')
# del n2v_feature
# gc.collect()
# print('==> data now size:', data.info(memory_usage='deep'))
# data = reduce_mem(data, tf_idf_feature)
# print('==> data now size:', data.info(memory_usage='deep'))

#################################################对特征后处理####################################################################
# 对类别特征labelencode并处理空值
# 1.fill nan dense_feature and do simple Transformation for dense features
# 如果loss初始大于1，可能是存在控制，或者数据类型不正确， label类型等
print('==>start process')  # 这里在运行的时候会多20g然后在下降
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
data[dense_features] = np.log(data[dense_features] + 1.0)

# 512维embedding方法二使用 已经确认不存在非空值，word2vec处理的特征需要在dense处理之后加
# for feat in sparse_features:
#     lbe = LabelEncoder()  # 将离散型的数据标签化
#     data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))  # 对数据进行归一化
dense_features += test_fea
data[dense_features] = mms.fit_transform(data[dense_features])  # 将dense数据归一化到(0,1)

dense_features += w2v_features
sparse_features += keyword_features
dense_features += tf_idf_feature

data[dense_features] = data[dense_features].fillna(0, )

data = reduce_mem(data, dense_features)
data = reduce_mem(data, sparse_features)
# dense_features += embed_featrure2
print('sparse feature:', sparse_features)
print('dense feature:', dense_features)
print('final data shape：', data.shape)
print('all_data占据内存约: {:.2f} GB'.format(data.memory_usage().sum() / (1024 ** 3)))
#################################################保存处理后的特征####################################################################
print('==> saveing data')
traning_days = 15
import json

# compressed_file = "../../training_data/train_data.feather"
data[data['date_'] < traning_days].reset_index().to_feather(TRAN_PATH + "train_data.feather", compression='zstd',
                                                            compression_level=2)
data[data['date_'] == 14].reset_index().to_feather(TRAN_PATH + "val_data.feather", compression='zstd',
                                                   compression_level=2)
data[data['date_'] == 15].reset_index().to_feather(TRAN_PATH + "test_data.feather", compression='zstd',
                                                   compression_level=2)
# data.to_feather(compressed_file,compression ='zstd',compression_level =2)

with open(os.path.join(TRAN_PATH, f"sparse_features.json"), "w") as op:
    json.dump(sparse_features, op)
with open(os.path.join(TRAN_PATH, f"dense_features.json"), "w") as op:
    json.dump(dense_features, op)
# if os.path.exists(os.path.join(TRAN_PATH, f"sparse_features.json")):
#     with open(os.path.join(TRAN_PATH, f"sparse_features.json")) as op:
#         features_columns = json.load(op)
#         print(features_columns)

del data
gc.collect()
print('==>save finished')
