import tensorflow as tf

print(tf.test.gpu_device_name())
print(tf.__version__)
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import sys
import gc
import random
from time import time
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.python.keras.utils import multi_gpu_model

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()
print(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../'))
# from evaluation import evaluate_uAUC
# sys.path.append(os.path.join(BASE_DIR, '../model'))
# from model.mmoe import MMOE
# from model.model import AutoInt
from deepctr.models.xdeepfm import xDeepFM
from tensorflow.python.keras.initializers import RandomNormal, Zeros, TruncatedNormal, RandomUniform
from tqdm import tqdm
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers
import signal
from tensorflow.python.keras.optimizers import Adam, Adagrad

print(tf.test.gpu_device_name())
print(tf.__version__)

'''
本地0.73
线上0.681

'''
# GPU相关设置
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
# target = ["favorite","forward", "comment", "follow"]
# target = ["follow", "comment"]
Tasks = ['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary']

BASE_PATH = '../../data/feather_data/'
FEA_PATH = "../../data/fea_data/"
chache_PATH = '../../data/cache/'
MODEL_PATH = '../../data/model/'
key2index = {}


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

    user_auc = float(total_auc) / size

    return user_auc


sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'device']
dense_features = ['videoplayseconds']
w2v_features = []
tf_idf_feature = []
test_fea = []
keyword_features = []
##################读取id特征 ##############################

print("==> Loading training data")
t0 = time()

# action_data = pd.read_feather(BASE_PATH + 'user_action.feather', columns=None, use_threads=True)
# feed = pd.read_feather(BASE_PATH + 'feed_info.feather', columns=None, use_threads=True)
# test_data = pd.read_feather(BASE_PATH + 'test_a.feather', columns=None, use_threads=True)
# 原始数据
# BASE_PATH = '../../../wbdc2021/data/wedata/'
action_data = pd.read_feather(BASE_PATH + 'user_action_just_fusai.feather')
feed = pd.read_feather(BASE_PATH + 'feed_info.feather')
test_data = pd.read_feather(BASE_PATH + 'wechat_algo_data2/test_a.csv')

t1 = time()
print('read_data_time:', t1 - t0)

# 合并数据集,统一处理
# test_data[ACTION_LIST] = 0
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
data.head()
print('==>data_base_fea_shape:', data.shape)

t2 = time()
print('read_base_data_cost_time:', t2 - t1)
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
# v5 加入n2v特征
print('==>add n2v feature')
n2v_feature = pd.read_pickle(FEA_PATH + 'v5_1_userid_authorid_n2v.pkl')

# print('==> n2v size:', n2v_feature.info(memory_usage='deep'))
# print('n2v_feature: {:.2f} GB'.format(n2v_feature.memory_usage().sum()/ (1024**3)))
print('==>merge n2v feature')
data = data.merge(n2v_feature, on=["userid"], how="left")
n2v_feature = n2v_feature.set_index(["userid"])
tf_idf_feature += n2v_feature.columns.to_list()
n2v_feature = reduce_mem(data, tf_idf_feature)  # [0:6]
del n2v_feature
gc.collect()
print('data shape：', data.shape)
print('==> data now size:', data.info(memory_usage='deep'))

# v5 加入deepwalk特征
n2v_feature = pd.read_feather(FEA_PATH + 'fea_user_deepwalk.feather', columns=None, use_threads=True)  # 原始复赛数据
n2v_feature = reduce_mem(n2v_feature, n2v_feature.columns.to_list())
# print('==> deepwalk size:', n2v_feature.info(memory_usage='deep'))
# print('n2v_feature: {:.2f} GB'.format(n2v_feature.memory_usage().sum()/ (1024**3)))
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

print('==>start process')
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
data[sparse_features] = data[sparse_features].fillna(0, )

data = reduce_mem(data, dense_features)
data = reduce_mem(data, sparse_features)

# dense_features += embed_featrure2
print('sparse feature:', sparse_features)
print('dense feature:', dense_features)
print('data.shape', data.shape)

# import json
# compressed_file = FEA_PATH + "train_data_just_fusai.feather"
# #compressed_file = "../../training_data/train_data.feather"
# data.to_feather(compressed_file,compression ='zstd',compression_level =2)
# print('save finished')
# outpath = '../../training_data/'

# with open(os.path.join(FEA_PATH, f"sparse_features_just_fusai.json"), "w") as op:
#     json.dump(sparse_features, op)
# with open(os.path.join(FEA_PATH, f"dense_features_just_fusai.json"), "w") as op:
#     json.dump(dense_features, op)
# if os.path.exists(os.path.join(FEA_PATH, f"sparse_features_just_fusai.json")):
#     with open(os.path.join(FEA_PATH, f"sparse_features_just_fusai.json")) as op:
#         features_columns = json.load(op)
#         print(features_columns)


# 前14天数据进行训练或者前15天数据进行训练,线上提交为15天

num_repeat = 1
fix_seed_repeat = 1
traning_days = 15
embedding_dim = 8
batch_size = 4096

print("==>load data")
# compressed_file = FEA_PATH + "train_data.feather"
# data = pd.read_feather(compressed_file, columns=None, use_threads=True)

# if os.path.exists(os.path.join(FEA_PATH, f"sparse_features.json")):
#     with open(os.path.join(FEA_PATH, f"sparse_features.json")) as op:
#         sparse_features = json.load(op)
# if os.path.exists(os.path.join(FEA_PATH, f"dense_features.json")):
#     with open(os.path.join(FEA_PATH, f"dense_features.json")) as op:
#         dense_features = json.load(op)
# print("==>load data finished")
train = data[data['date_'] < traning_days]  # 提交请使用15天全部样本
val = data[data['date_'] == 14]  # 第14天样本作为验证集
test = data[data['date_'] == 15]  # 第14天样本作为验证集

# 2.count #unique features for each sparse field,and record dense features field name

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=embedding_dim)
                          for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]

linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(dnn_feature_columns)

del data
gc.collect()
# dnn_feature_columns.append(user_embedding)

# 3.generate input data for model
train_model_input = {name: train[name] for name in feature_names}
val_model_input = {name: val[name] for name in feature_names}
userid_list = val['userid'].astype(str).tolist()
test_model_input = {name: test[name] for name in feature_names}

train_labels = [train[y].values for y in target]
val_labels = [val[y].values for y in target]

del train
del val
del test
gc.collect()
print("==>split data finished")

print('training start')
from tensorflow.python.keras.utils import multi_gpu_model

tf.keras.backend.clear_session()
# 4.Define Model,train,predict and evaluate
# model = AutoInt(linear_feature_columns,dnn_feature_columns,att_layer_num=4, att_embedding_size=16, att_head_num=4)
# model.compile('adam','binary_crossentropy',metrics=[tf.keras.metrics.AUC(name='auc')])


# train_model = deepfm(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(1024, 512, 128),seed=2021)# task_dnn_units= (1024, 512, 128), #可能比例越低的目标需要更深的网络来提取
# train_model = multi_gpu_model(train_model, gpus=2)
# train_model.compile("adagrad", loss='binary_crossentropy')
# train_model.compile('adam', loss='binary_crossentropy')
# loss_weights = []
# #train_model.compile("adadelta", loss='binary_crossentropy')
# train_model.compile(Adagrad(0.05), loss='binary_crossentropy')
# warm_up_lr = WarmUpLearningRateScheduler(batch_size, init_lr=0.02)#Adagrad

all_uAUC = []
seed = [16168, 2021, 22231, 28294]
i = random.randint(1, 20)
seed.append(i)
x = random.randint(1, 20)
seed.append(x)
for z in seed:
    print('############# seed ' + str(z * 2021) + '#######################')
    for i, y in enumerate(target):
        if (y == 'read_comment' or y == 'like' or y == 'click_avatar'):
            epoch = 4
            train_model = xDeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(1024, 512, 128),
                                  seed=z)
            train_model.compile(Adagrad(0.05), loss='binary_crossentropy')
            warm_up_lr = WarmUpLearningRateScheduler(batch_size, init_lr=0.02)
        elif (y == 'click_avatar'):
            epoch = 4
            train_model = xDeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(2048, 1024, 512),
                                  seed=z)  # task_dnn_units= (1024, 512, 128), #可能比例越低的目标需要更深的网络来提
            train_model.compile(Adagrad(0.01), loss='binary_crossentropy')
            warm_up_lr = WarmUpLearningRateScheduler(batch_size, init_lr=0.04)
        elif (y == 'favorite'):
            epoch = 1
            train_model = xDeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(2048, 1024, 512),
                                  seed=z)  # task_dnn_units= (1024, 512, 128), #可能比例越低的目标需要更深的网络来提
            train_model.compile(Adagrad(0.01), loss='binary_crossentropy')
            warm_up_lr = WarmUpLearningRateScheduler(batch_size, init_lr=0.04)
        else:
            epoch = 1
            train_model = xDeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(1024, 512, 128),
                                  seed=z)
            train_model.compile(Adagrad(0.05), loss='binary_crossentropy')
            warm_up_lr = WarmUpLearningRateScheduler(batch_size, init_lr=0.02)

        # for epoch in range(epochs):

        print('epochs:', epoch)
        history = train_model.fit(train_model_input, train_labels[i],
                                  batch_size=batch_size, epochs=epoch, verbose=1, callbacks=[warm_up_lr])  #

        val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
        # uAUC = evaluate_uAUC(val_labels, val_pred_ans, userid_list, target)
        val_uauc = uAUC(val_labels[i], val_pred_ans, userid_list)
        print('==> ' + y + str(val_uauc))
        all_uAUC.append(val_uauc)

        # pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 20)
        train_model.save_weights(MODEL_PATH + 'xdeepfm_allfeature_' + str(z) + '_' + str(val_uauc) + '_' + y + '.h5')
        # train_model.load_weights(BASE_PATH + 'model/mmoe.h5')

        print('read predict')
        pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 20)
        # SUB_PATH = '../../data/submission/'
        # 5.生成提交文件
        # for i, action in enumerate(target):
        submit[y] = pred_ans
    print('uauc:',
          all_uAUC[0] * (4 / 13) + all_uAUC[1] * (3 / 13) + all_uAUC[2] * (2 / 13) + all_uAUC[3] * (1 / 13) + all_uAUC[
              4] * (1 / 13) + all_uAUC[5] * (1 / 13) + all_uAUC[6] * (1 / 13))
    # zlx = all_uAUC[0]*(4/13)+all_uAUC[1]*(3/13)+all_uAUC[2]*(2/13)+all_uAUC[3]*(1/13)+all_uAUC[4]*(1/13)+all_uAUC[5]*(1/13)+all_uAUC[6]*(1/13)
    SUB_PATH = './result/'
    try:
        submit.to_csv(SUB_PATH + 'xdeepfm_sub' + str(z) + '.csv',
                      index=None, float_format='%.6f')
    except:
        submit.to_csv(SUB_PATH + 'sub_%.6f.csv' % (uAUC), index=None, float_format='%.6f')
    submit.head()

