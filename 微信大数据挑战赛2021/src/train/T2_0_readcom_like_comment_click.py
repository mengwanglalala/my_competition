import tensorflow as tf

print(tf.test.gpu_device_name())
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import gc
import json
from time import time
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA

# from tensorflow.python.keras.utils import multi_gpu_model
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
import signal

# GPU相关设置
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 设置GPU按需增长
config = tf.ConfigProto()
# config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# sess = tf.compat.v1.Session(config=config)


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


#################################################划分数据集准备训练####################################################################
# 前14天数据进行训练或者前15天数据进行训练,线上提交为15天
def main(argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # 设置GPU按需增长
    config = tf.ConfigProto()
    # config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    epochs = 4
    num_repeat = 1
    fix_seed_repeat = 1
    embedding_dim = 32
    batch_size = 4096

    SEED = int(argv[1])
    print('seed:', SEED)

    # 用于构造特征的字段列表
    FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
    ACTION_LIST = ["read_comment", "like", "click_avatar", "favorite", "forward", "comment", "follow"]
    target = ["read_comment", "like", "click_avatar", "favorite", "forward", "comment", "follow"]
    Tasks = ['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary']
    BASE_PATH = '../../data/feather_data/'
    FEA_PATH = "../../data/fea_data/"
    chache_PATH = '../../data/cache/'
    MODEL_PATH = '../../data/model/'
    TRAN_PATH = '../../data/train_data/'
    key2index = {}

    #     target = ["read_comment", "like", "click_avatar","favorite","forward", "comment", "follow"]
    #     Tasks = ['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary']
    #     NUM_TASK = 7

    target = ["read_comment", "like", "click_avatar", "comment"]
    Tasks = ['binary', 'binary', 'binary', 'binary']
    NUM_TASK = 4

    print("==>load data")

    train = pd.read_feather(TRAN_PATH + "train_data.feather", columns=None, use_threads=True)  # 提交请使用15天全部样本
    val = pd.read_feather(TRAN_PATH + "val_data.feather", columns=None, use_threads=True)  # 第14天样本作为验证集
    test = pd.read_feather(TRAN_PATH + "test_data.feather", columns=None, use_threads=True)  # 第14天样本作为验证集

    if os.path.exists(os.path.join(TRAN_PATH, f"sparse_features.json")):
        with open(os.path.join(TRAN_PATH, f"sparse_features.json")) as op:
            sparse_features = json.load(op)
    if os.path.exists(os.path.join(TRAN_PATH, f"dense_features.json")):
        with open(os.path.join(TRAN_PATH, f"dense_features.json")) as op:
            dense_features = json.load(op)
    print("==>load data finished")

    # print('sparse feature:',sparse_features)
    print('dense feature:', dense_features)
    print('import data shape：', train.shape, val.shape, test.shape)

    # 2.count #unique features for each sparse field,and record dense features field name
    # 整理vocabulary_size需要改train[feat].max() + 1
    # vocabulary size需要注意！！！！
    # fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=300000, embedding_dim=embedding_dim)
    #                           for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]

    fixlen_feature_columns = [SparseFeat('userid', vocabulary_size=300000, embedding_dim=32),
                              SparseFeat('feedid', vocabulary_size=300000, embedding_dim=32),
                              SparseFeat('authorid', vocabulary_size=100000, embedding_dim=16),
                              SparseFeat('bgm_song_id', vocabulary_size=100000, embedding_dim=16),
                              SparseFeat('bgm_singer_id', vocabulary_size=100000, embedding_dim=16),
                              SparseFeat('device', vocabulary_size=3, embedding_dim=4)
                              ] + [DenseFeat(feat, 1) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(dnn_feature_columns)

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
    #################################################构建模型并训练####################################################################
    print('training start')
    from tensorflow.python.keras.utils import multi_gpu_model

    tf.keras.backend.clear_session()
    # 4.Define Model,train,predict and evaluate
    train_model = MMOE(dnn_feature_columns, num_tasks=NUM_TASK, expert_dim=28, dnn_hidden_units=(2048, 1024, 512, 256),
                       tasks=Tasks,
                       seed=SEED)  # task_dnn_units= (1024, 512, 128), #可能比例越低的目标需要更深的网络来提取

    train_model.compile(Adagrad(0.01), loss='binary_crossentropy')  # Adagrad
    warm_up_lr = WarmUpLearningRateScheduler(batch_size, init_lr=0.02)
    # train_model = multi_gpu_model(train_model, gpus=2)
    # train_model.compile("adagrad", loss='binary_crossentropy')
    # train_model.compile("adam", loss='binary_crossentropy')  # loss_weights = []

    # train_model.compile("adadelta", loss='binary_crossentropy')
    # train_model.compile(Adagrad(0.001), loss='binary_crossentropy')#Adagrad
    # print(train_model.summary())
    all_uAUC = []
    for epoch in range(epochs):
        print('epoch:', epoch)
        history = train_model.fit(train_model_input, train_labels,
                                  batch_size=batch_size, epochs=1, verbose=1, use_multiprocessing=True,
                                  callbacks=[warm_up_lr])

        val_pred_ans = train_model.predict(val_model_input, batch_size=batch_size * 4)
        uAUC = evaluate_uAUC(val_labels, val_pred_ans, userid_list, target)
        all_uAUC.append(uAUC)
    train_model.save_weights(MODEL_PATH + 'mmoe_seed_' + str(SEED) + '_readcom_like_com_click_' + '.h5')
    # pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 20)
    # train_model.load_weights(MODEL_PATH + 'mmoe' +'.h5')

    #################################################导出预测结果####################################################################
    print('read predict')
    pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 20)
    SUB_PATH = './result/'
    BASE_PATH = '../../../wbdc2021/data/wedata/'
    test_data = pd.read_csv(BASE_PATH + 'wechat_algo_data2/test_a.csv')
    submit = test_data[['userid', 'feedid']]
    # 5.生成提交文件
    for i, action in enumerate(target):
        submit[action] = pred_ans[i]
    try:
        submit.to_csv(SUB_PATH + 'targ_readcom_like_com_click_seed_' + str(SEED) + '_sub_%.6f_%.6f_%.6f_%.6f.csv' % (
        all_uAUC[0], all_uAUC[1], all_uAUC[2], all_uAUC[3]),
                      index=None, float_format='%.6f')
    except:
        submit.to_csv(SUB_PATH + 'sub_%.6f.csv' % (uAUC), index=None, float_format='%.6f')
    # submit.to_csv(SUB_PATH + target[0]+ '_' + target[1]+ '_' +'sub_%.6f.csv' % (uAUC), index=None, float_format='%.6f')
    submit.head()
    print('training finished')

    os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    tf.app.run(main)
