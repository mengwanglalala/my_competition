# coding: utf-8
import tensorflow as tf

print(tf.test.gpu_device_name())
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import gc
import signal
import time
import json
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
from deepctr.models import AutoInt
from deepctr.models.deepfm import DeepFM
from deepctr.models.xdeepfm import xDeepFM
from tensorflow.python.keras.initializers import RandomNormal, Zeros, TruncatedNormal, RandomUniform
from tqdm import tqdm
from tensorflow.python.keras.optimizers import Adam, Adagrad

print(tf.test.gpu_device_name())
print(tf.__version__)
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers

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
Tasks = ['binary', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary']
NUM_TASK = 7

BASE_PATH = '../../wbdc2021/data/wedata/'
FEA_PATH = '../data/fea_data/'
SUMIT_DIR = '../data/submission/'
model_file = '../data/model/mmoe.h5'
TRAN_PATH = '../data/train_data/'
MODEL_PATH = '../data/model/'


# target = ["read_comment", "comment"]
# Tasks = ['binary', 'binary']
# NUM_TASK = 2
# test_path = '../wbdc2021/data/wedata/wechat_algo_data2/test_a.csv'
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


#################################################merge数据（需要重写）####################################################################
def get_test_data(test_path, dense_features_input, sparse_features):
    print(test_path)
    data = pd.read_csv(test_path)
    feed = pd.read_csv(BASE_PATH + 'wechat_algo_data2/feed_info.csv')
    data["date_"] = 15

    feed[["bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        feed[["bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    feed['bgm_song_id'] = feed['bgm_song_id'].astype('int64')
    feed['bgm_singer_id'] = feed['bgm_singer_id'].astype('int64')

    data = data.merge(
        feed[['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']],
        how='left',
        on='feedid')
    dense_features = ['videoplayseconds']
    tf_idf_feature = []

    # v0 统计特征
    print('add statis feature')
    statis_fea = pd.read_feather(FEA_PATH + 'v0_statis_5_day_featrue_all_data_drop_duplicate_first.feather',
                                 columns=None, use_threads=True)  # 10
    print('==> statis feature size:', statis_fea.info(memory_usage='deep'))
    cols = ['userid_in_feedid_nunique',
            'feedid_in_userid_nunique', 'userid_in_authorid_nunique', 'authorid_in_userid_nunique',
            'userid_authorid_count',
            'userid_in_authorid_count_prop', 'authorid_in_userid_count_prop', 'videoplayseconds_in_userid_mean',
            'videoplayseconds_in_authorid_mean', 'feedid_in_authorid_nunique']
    # statis_fea = reduce_mem(statis_fea, cols)
    print('statis_fea: {:.2f} GB'.format(statis_fea.memory_usage().sum() / (1024 ** 3)))
    data = data.merge(statis_fea, on=["userid", "feedid", "date_"], how="left")
    dense_features += cols
    del statis_fea
    gc.collect()
    print('==> data now size:', data.info(memory_usage='deep'))

    # # v1 512维度embedding
    print('add statis 512embedding')
    embedding_512_feature = pd.read_feather(FEA_PATH + 'v1_feed_embeddings_process.feather', columns=None,
                                            use_threads=True)
    data = data.merge(embedding_512_feature, on=["feedid"], how="left")
    embedding_512_feature = embedding_512_feature.set_index(["feedid"])
    tf_idf_feature += embedding_512_feature.columns.to_list()  # [0:6]
    del embedding_512_feature
    gc.collect()

    # v5 加入n2v特征
    print('==>add n2v feature')
    n2v_feature = pd.read_pickle(FEA_PATH + 'v5_1_userid_authorid_n2v.pkl')
    # n2v_feature = reduce_mem(n2v_feature,  n2v_feature.columns.to_list())
    print('==> n2v size:', n2v_feature.info(memory_usage='deep'))
    print('n2v_feature: {:.2f} GB'.format(n2v_feature.memory_usage().sum() / (1024 ** 3)))
    n2v_feature = n2v_feature.set_index(["userid"])
    tf_idf_feature += n2v_feature.columns.to_list()  # [0:6]
    print('==>merge n2v feature')
    data = data.merge(n2v_feature, on=["userid"], how="left")
    del n2v_feature
    gc.collect()
    print('data shape：', data.shape)
    print('==> data now size:', data.info(memory_usage='deep'))

    n2v_feature = pd.read_feather(FEA_PATH + 'fea_user_deepwalk.feather', columns=None, use_threads=True)  # 原始复赛数据
    # n2v_feature = reduce_mem(n2v_feature,  n2v_feature.columns.to_list())
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
    # data = reduce_mem(data, tf_idf_feature)
    print('==> data now size:', data.info(memory_usage='deep'))

    #################################################对特征后处理####################################################################
    print('==>start process')  # 这里在运行的时候会多20g然后在下降
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    data[dense_features] = np.log(data[dense_features] + 1.0)

    # 512维embedding方法二使用 已经确认不存在非空值，word2vec处理的特征需要在dense处理之后加
    # for feat in sparse_features:
    #     lbe = LabelEncoder()  # 将离散型的数据标签化
    #     data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))  # 对数据进行归一化
    data[dense_features] = mms.fit_transform(data[dense_features])  # 将dense数据归一化到(0,1)

    dense_features += tf_idf_feature

    data[dense_features] = data[dense_features].fillna(0, )

    assert dense_features_input == dense_features

    #     data = reduce_mem(data, dense_features)
    #     data = reduce_mem(data, sparse_features)
    # dense_features += embed_featrure2
    print('sparse feature:', sparse_features)
    print('dense feature:', dense_features)
    print('final data shape：', data.shape)
    print('all_data占据内存约: {:.2f} GB'.format(data.memory_usage().sum() / (1024 ** 3)))

    # 测试用, 提交文件的时候注意注释掉
    # data =  pd.read_feather(TRAN_PATH + "val_data.feather", columns=None, use_threads=True)# 第14天样本作为验证集
    print('test shape:', data.head())

    return data


def get_feature_columns():
    #################################################加载数据####################################################################
    print("==>load data")
    if os.path.exists(os.path.join(TRAN_PATH, f"sparse_features.json")):
        with open(os.path.join(TRAN_PATH, f"sparse_features.json")) as op:
            sparse_features = json.load(op)
    if os.path.exists(os.path.join(TRAN_PATH, f"dense_features.json")):
        with open(os.path.join(TRAN_PATH, f"dense_features.json")) as op:
            dense_features = json.load(op)
    print("==>load data finished")

    #     print('sparse feature:',sparse_features)
    #     print('dense feature:',dense_features)

    # 2.count #unique features for each sparse field,and record dense features field name
    #     fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=300000, embedding_dim=embedding_dim) #这里有问题啊
    #                               for feat in sparse_features] + [DenseFeat(feat, 1) for feat in dense_features]

    fixlen_feature_columns = [SparseFeat('userid', vocabulary_size=300000, embedding_dim=32),
                              SparseFeat('feedid', vocabulary_size=300000, embedding_dim=32),
                              SparseFeat('authorid', vocabulary_size=100000, embedding_dim=16),
                              SparseFeat('bgm_song_id', vocabulary_size=100000, embedding_dim=16),
                              SparseFeat('bgm_singer_id', vocabulary_size=100000, embedding_dim=16),
                              SparseFeat('device', vocabulary_size=3, embedding_dim=4)
                              ] + [DenseFeat(feat, 1) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    return dnn_feature_columns, linear_feature_columns, dense_features, sparse_features


def main(argv):
    # 前14天数据进行训练或者前15天数据进行训练,线上提交为15天
    epochs = 4
    num_repeat = 1
    fix_seed_repeat = 1
    batch_size = 4096

    # mmoe多种子融合 单模707
    seed_model_deeper_list = ['mmoe_deeper_seed_67.h5', 'mmoe_deeper_seed_105.h5', 'mmoe_deeper_seed_9745.h5',
                              'mmoe_deeper_seed_1234.h5',
                              'mmoe_deeper_seed_875.h5', 'mmoe_deeper_seed_643.h5', 'mmoe_deeper_seed_456.h5',
                              'mmoe_deeper_seed_120.h5']

    # mmoe分开建模融合 单模组合707
    split_model_list0 = [['mmoe_seed_456_readcom_like_com_click_.h5', 'mmoe_seed_789_readcom_like_com_click_.h5',
                          'mmoe_seed_123_readcom_like_com_click_.h5', 'mmoe_seed_45478_readcom_like_com_click_.h5'],
                         ['mmoe_seed_456_like_favor_forward.h5', 'mmoe_seed_123_like_favor_forward.h5',
                          'mmoe_seed_789_like_favor_forward.h5', 'mmoe_seed_45478_like_favor_forward.h5'],
                         ['mmoe_seed_456_click_follow.h5', 'mmoe_seed_789_click_follow.h5',
                          'mmoe_seed_123_click_follow.h5', 'mmoe_seed_45478_click_follow.h5']]

    # mmoe多种子融合 单模704
    seed_model_list = ['mmoe_seed_47.h5', 'mmoe_seed_1998.h5', 'mmoe_seed_2021.h5', 'mmoe_seed_1010.h5',
                       'mmoe_seed_4096.h5', 'mmoe_seed_6666.h5', 'mmoe_seed_123.h5', 'mmoe_seed_520.h5']

    # mmoe分开建模融合 单模组合705
    split_model_list = [['mmoe_seed_2021_readcom_like_com_click_.h5', 'mmoe_seed_22_readcom_like_com_click_.h5',
                         'mmoe_seed_182_readcom_like_com_click_.h5'],
                        ['mmoe_seed_2021_like_favor_forward.h5', 'mmoe_seed_22_like_favor_forward.h5',
                         'mmoe_seed_182_like_favor_forward.h5'],
                        ['mmoe_seed_2021_click_follow.h5', 'mmoe_seed_22_click_follow.h5',
                         'mmoe_seed_182_click_follow.h5']]
    #     split_model_list = [['mmoe_seed_22_readcom_like_com_click_.h5'],
    #                        ['mmoe_seed_22_like_favor_forward.h5'],
    #                        ['mmoe_seed_22_click_follow.h5']]
    ########################################加载特征名，测试数据merge特征#########################################
    t = time.time()
    dnn_feature_columns, linear_feature_columns, dense_features, sparse_features = get_feature_columns()
    stage = argv[1]
    print('Stage: %s' % stage)
    test_path = ''
    if len(argv) == 3:
        test_path = argv[2]
        t1 = time.time()
        test_data = get_test_data(test_path, dense_features, sparse_features)
        submit = test_data[['userid', 'feedid']]
        feature_names = get_feature_names(dnn_feature_columns)
        test_model_input = {name: test_data[name] for name in feature_names}
        print('Get test input cost: %.4f s' % (time.time() - t1))

    ########################################建立模型并预测（模型融合改这里）#########################################
    print('bulid model')
    # from tensorflow.python.keras.utils import multi_gpu_model

    # train_model.load_weights(BASE_PATH + 'model/mmoe' + '_reapeat_' + str(j) + '_seed_'+ str(i)+'.h5')

    if stage == "submit" and test_path:
        print('mmoe多种子融合707')
        ########################################mmoe多种子融合(707)#########################################
        tf.keras.backend.clear_session()
        train_model = MMOE(dnn_feature_columns, num_tasks=NUM_TASK, expert_dim=28,
                           dnn_hidden_units=(2048, 1024, 512, 256),
                           tasks=Tasks,
                           seed=num_repeat * 400)
        pred_ans_temp = []
        # 加载模型并进行预测
        for model_file2 in seed_model_deeper_list:
            train_model.load_weights(MODEL_PATH + model_file2)
            pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 4)
            pred_ans_temp.append(pred_ans)

        # mmoe多种子fusion
        muti_deeper_mmoe_fusion = submit[['userid', 'feedid']]
        for i, action in enumerate(target):
            for j in range(len(pred_ans_temp)):
                if j == 0:
                    ans_temp = pred_ans_temp[j][i] / len(pred_ans_temp)
                else:
                    ans_temp += pred_ans_temp[j][i] / len(pred_ans_temp)
            muti_deeper_mmoe_fusion[action] = ans_temp
        del ans_temp
        del pred_ans
        del train_model
        gc.collect()
        print('muti_mmoe_fusion:', muti_deeper_mmoe_fusion.head())
        print('mmoe分成三部分分开建模707')
        ########################################mmoe分成三部分分开建模(707)#########################################
        #         # 分开建模融合
        #         # 第一部分
        #         split_submit1_0 = submit[['userid', 'feedid']]
        #         pred_ans_temp = []
        #         tf.keras.backend.clear_session()
        #         train_model_split0 = MMOE(dnn_feature_columns, num_tasks=4, expert_dim=28, dnn_hidden_units=(2048, 1024, 512, 256),
        #                                   tasks=['binary', 'binary', 'binary', 'binary'],
        #                                   seed=num_repeat * 400)
        #         print(len(split_model_list0[0]))
        #         for i in range(len(split_model_list0[0])):
        #             # target = ["read_comment", "like", "click_avatar", "comment"]
        #             print(split_model_list0[0][i])
        #             train_model_split0.load_weights(MODEL_PATH + split_model_list0[0][i])
        #             pred_split0_ans = train_model_split0.predict(test_model_input, batch_size=batch_size * 4)
        #             pred_ans_temp.append(pred_split0_ans)

        #         for i, action in enumerate(["read_comment", "like", "click_avatar", "comment"]):
        #             for j in range(len(pred_ans_temp)):
        #                 if j == 0:
        #                     ans_temp = pred_ans_temp[j][i] / len(pred_ans_temp)
        #                 else:
        #                     ans_temp += pred_ans_temp[j][i] / len(pred_ans_temp)
        #             split_submit1_0[action] = ans_temp
        #         del pred_split0_ans
        #         del train_model_split0
        #         gc.collect()

        #         # 第二部分
        #         split_submit2_0 = submit[['userid', 'feedid']]
        #         pred_ans_temp = []
        #         train_model_split0 = MMOE(dnn_feature_columns, num_tasks=3, expert_dim=28, dnn_hidden_units=(2048, 1024, 512, 256),
        #                                   tasks=['binary', 'binary', 'binary'],
        #                                   seed=num_repeat * 400)
        #         for i in range(len(split_model_list0[0])):
        #             # target = [ "like", "favorite","forward"]
        #             train_model_split0.load_weights(MODEL_PATH + split_model_list0[1][i])
        #             pred_split0_ans = train_model_split0.predict(test_model_input, batch_size=batch_size * 4)
        #             pred_ans_temp.append(pred_split0_ans)
        #         for i, action in enumerate(["like", "favorite", "forward"]):
        #             for j in range(len(pred_ans_temp)):
        #                 if j == 0:
        #                     ans_temp = pred_ans_temp[j][i] / len(pred_ans_temp)
        #                 else:
        #                     ans_temp += pred_ans_temp[j][i] / len(pred_ans_temp)
        #             split_submit2_0[action] = ans_temp
        #         del pred_split0_ans
        #         del train_model_split0
        #         gc.collect()

        #         # 第三部分
        #         split_submit3_0 = submit[['userid', 'feedid']]
        #         pred_ans_temp = []
        #         train_model_split0 = MMOE(dnn_feature_columns, num_tasks=2, expert_dim=28, dnn_hidden_units=(2048, 1024, 512, 256),
        #                                   tasks=['binary', 'binary'],
        #                                   seed=num_repeat * 400)
        #         for i in range(len(split_model_list0[0])):
        #             # target = ["click_avatar", "follow"]
        #             train_model_split0.load_weights(MODEL_PATH + split_model_list0[2][i])
        #             pred_split0_ans = train_model_split0.predict(test_model_input, batch_size=batch_size * 4)
        #             pred_ans_temp.append(pred_split0_ans)
        #         for i, action in enumerate(["click_avatar", "follow"]):
        #             for j in range(len(pred_ans_temp)):
        #                 if j == 0:
        #                     ans_temp = pred_ans_temp[j][i] / len(pred_ans_temp)
        #                 else:
        #                     ans_temp += pred_ans_temp[j][i] / len(pred_ans_temp)
        #             split_submit3_0[action] = ans_temp

        #         del pred_split0_ans
        #         del train_model_split0
        #         gc.collect()

        #         split_submit0 = split_submit1_0[['userid', 'feedid', "read_comment", "comment"]]
        #         split_submit0[["like", "favorite", "forward"]] = split_submit2_0[["like", "favorite", "forward"]]
        #         split_submit0[["click_avatar", "follow"]] = split_submit3_0[["click_avatar", "follow"]]
        #         # 把预测了的都用上
        #         split_submit0["like"] = (split_submit0["like"] + split_submit1_0["like"]) / 2
        #         split_submit0["click_avatar"] = (split_submit0["click_avatar"] + split_submit1_0["click_avatar"]) / 2
        #         print('split_submit', split_submit0.head())

        #         print('mmoe多种子融合704')
        ########################################mmoe多种子融合(704)#########################################
        tf.keras.backend.clear_session()
        # 4.Define Model,train,predict and evaluate
        train_model = MMOE(dnn_feature_columns, num_tasks=NUM_TASK, expert_dim=28, dnn_hidden_units=(1024, 512, 128),
                           tasks=Tasks,
                           seed=num_repeat * 400)

        train_model.compile(Adagrad(0.01), loss='binary_crossentropy')  # Adagrad
        pred_ans_temp = []
        # 加载模型并进行预测
        for model_file2 in seed_model_list:
            train_model.load_weights(MODEL_PATH + model_file2)
            pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 4)
            pred_ans_temp.append(pred_ans)

        # mmoe多种子fusion
        muti_mmoe_fusion = submit[['userid', 'feedid']]
        for i, action in enumerate(target):
            for j in range(len(pred_ans_temp)):
                if j == 0:
                    ans_temp = pred_ans_temp[j][i] / len(pred_ans_temp)
                else:
                    ans_temp += pred_ans_temp[j][i] / len(pred_ans_temp)
            muti_mmoe_fusion[action] = ans_temp
        del ans_temp
        del pred_ans
        del train_model
        gc.collect()
        print('muti_mmoe_fusion:', muti_mmoe_fusion.head())
        ########################################mmoe分成三部分分开建模(705)#########################################
        # 分开建模融合
        # 第一部分
        split_submit1 = submit[['userid', 'feedid']]
        pred_ans_temp = []
        tf.keras.backend.clear_session()
        train_model_split0 = MMOE(dnn_feature_columns, num_tasks=4, expert_dim=28, dnn_hidden_units=(1024, 512, 128),
                                  tasks=['binary', 'binary', 'binary', 'binary'],
                                  seed=num_repeat * 400)
        print(len(split_model_list[0]))
        for i in range(len(split_model_list[0])):
            # target = ["read_comment", "like", "click_avatar", "comment"]
            print(split_model_list[0][i])
            train_model_split0.load_weights(MODEL_PATH + split_model_list[0][i])
            pred_split0_ans = train_model_split0.predict(test_model_input, batch_size=batch_size * 4)
            pred_ans_temp.append(pred_split0_ans)

        for i, action in enumerate(["read_comment", "like", "click_avatar", "comment"]):
            for j in range(len(pred_ans_temp)):
                if j == 0:
                    ans_temp = pred_ans_temp[j][i] / len(pred_ans_temp)
                else:
                    ans_temp += pred_ans_temp[j][i] / len(pred_ans_temp)
            split_submit1[action] = ans_temp
        del pred_split0_ans
        del train_model_split0
        gc.collect()

        # 第二部分
        split_submit2 = submit[['userid', 'feedid']]
        pred_ans_temp = []
        train_model_split0 = MMOE(dnn_feature_columns, num_tasks=3, expert_dim=28, dnn_hidden_units=(1024, 512, 128),
                                  tasks=['binary', 'binary', 'binary'],
                                  seed=num_repeat * 400)
        for i in range(len(split_model_list[0])):
            # target = [ "like", "favorite","forward"]
            train_model_split0.load_weights(MODEL_PATH + split_model_list[1][i])
            pred_split0_ans = train_model_split0.predict(test_model_input, batch_size=batch_size * 4)
            pred_ans_temp.append(pred_split0_ans)
        for i, action in enumerate(["like", "favorite", "forward"]):
            for j in range(len(pred_ans_temp)):
                if j == 0:
                    ans_temp = pred_ans_temp[j][i] / len(pred_ans_temp)
                else:
                    ans_temp += pred_ans_temp[j][i] / len(pred_ans_temp)
            split_submit2[action] = ans_temp
        del pred_split0_ans
        del train_model_split0
        gc.collect()

        # 第三部分
        split_submit3 = submit[['userid', 'feedid']]
        pred_ans_temp = []
        train_model_split0 = MMOE(dnn_feature_columns, num_tasks=2, expert_dim=28, dnn_hidden_units=(1024, 512, 128),
                                  tasks=['binary', 'binary'],
                                  seed=num_repeat * 400)
        for i in range(len(split_model_list[0])):
            # target = ["click_avatar", "follow"]
            train_model_split0.load_weights(MODEL_PATH + split_model_list[2][i])
            pred_split0_ans = train_model_split0.predict(test_model_input, batch_size=batch_size * 4)
            pred_ans_temp.append(pred_split0_ans)
        for i, action in enumerate(["click_avatar", "follow"]):
            for j in range(len(pred_ans_temp)):
                if j == 0:
                    ans_temp = pred_ans_temp[j][i] / len(pred_ans_temp)
                else:
                    ans_temp += pred_ans_temp[j][i] / len(pred_ans_temp)
            split_submit3[action] = ans_temp

        del pred_split0_ans
        del train_model_split0
        gc.collect()

        split_submit = split_submit1[['userid', 'feedid', "read_comment", "comment"]]
        split_submit[["like", "favorite", "forward"]] = split_submit2[["like", "favorite", "forward"]]
        split_submit[["click_avatar", "follow"]] = split_submit3[["click_avatar", "follow"]]
        # 把预测了的都用上
        split_submit["like"] = (split_submit["like"] + split_submit1["like"]) / 2
        split_submit["click_avatar"] = (split_submit["click_avatar"] + split_submit1["click_avatar"]) / 2
        print('split_submit', split_submit.head())

        ########################################最终所有模型融合#########################################
        # target = ["read_comment", "like", "click_avatar","favorite","forward", "comment", "follow"]
        wm_sub = submit[['userid', 'feedid']]
        #         for i, action in enumerate(target):#0.714974
        #             wm_sub[action] = (3 * muti_deeper_mmoe_fusion[action] + 1*split_submit0[action] + 2 * muti_mmoe_fusion[action])/6.0 #+ 0.2 * split_submit[action]
        for i, action in enumerate(target):
            wm_sub[action] = (3 * muti_deeper_mmoe_fusion[action] + 1 * split_submit[action] + 2 * muti_mmoe_fusion[
                action]) / 6.0  # + 0.2 *split_submit0[action]
        #         #######################################AutoInt(1)#############################################
        #         fixlen_feature_columns = [SparseFeat('userid', vocabulary_size=300000, embedding_dim=8),
        #                               SparseFeat('feedid', vocabulary_size=300000, embedding_dim=8),
        #                               SparseFeat('authorid', vocabulary_size=100000, embedding_dim=8),
        #                               SparseFeat('bgm_song_id', vocabulary_size=100000, embedding_dim=8),
        #                               SparseFeat('bgm_singer_id', vocabulary_size=100000, embedding_dim=8),
        #                               SparseFeat('device', vocabulary_size=3, embedding_dim=8)
        #                               ] + [DenseFeat(feat, 1) for feat in dense_features]

        #         dnn_feature_columns = fixlen_feature_columns
        #         linear_feature_columns = fixlen_feature_columns
        #         tf.keras.backend.clear_session()
        #         split_submit4 = submit[['userid', 'feedid']]
        #         for y in target:
        #             print(y)
        #             if y == 'comment':
        #                 train_model = AutoInt(linear_feature_columns,dnn_feature_columns,att_layer_num=6, att_embedding_size=8, att_head_num=8,dnn_hidden_units=(2048, 1024, 512,128),seed=2021)
        #                 train_model.load_weights(MODEL_PATH + 'AutoInt_2021_' + y + '.h5')
        #             else:
        #                 train_model = AutoInt(linear_feature_columns,dnn_feature_columns,att_layer_num=6, att_embedding_size=8, att_head_num=6,dnn_hidden_units=(1024, 512,128),seed=2021)
        #                 train_model.load_weights(MODEL_PATH + 'AutoInt_2021_' + y + '.h5')
        #             pre = train_model.predict(test_model_input,batch_size=batch_size * 20)
        #             split_submit4[y] = pre
        #         print(split_submit4.head())
        # # #         ########################################AutoInt(2)#############################################
        #         tf.keras.backend.clear_session()
        #         split_submit5 = submit[['userid', 'feedid']]
        #         for y in target:
        #             if (y=='read_comment'or y=='like' or y=='click_avatar'):
        #                 train_model = AutoInt(linear_feature_columns,dnn_feature_columns,att_layer_num=6, att_embedding_size=8, att_head_num=6,dnn_hidden_units=(1024, 512,128),seed=16168)
        #                 train_model.load_weights(MODEL_PATH + 'AutoInt_16168_' + y + '.h5')
        #             elif(y=='click_avatar'):
        #                 train_model = AutoInt(linear_feature_columns,dnn_feature_columns,att_layer_num=6, att_embedding_size=8, att_head_num=8,dnn_hidden_units=(2048, 1024, 512),seed=16168)  # task_dnn_units= (1024, 512, 128), #可能比例越低的目标需要更深的网络来提
        #                 train_model.load_weights(MODEL_PATH + 'AutoInt_16168_' + y + '.h5')
        #             elif(y=='favorite'):
        #                 train_model = AutoInt(linear_feature_columns,dnn_feature_columns,att_layer_num=6, att_embedding_size=8, att_head_num=6,dnn_hidden_units=(2048, 1024, 512),seed=16168)
        #                 train_model.load_weights(MODEL_PATH + 'AutoInt_16168_' + y + '.h5')
        #             else:
        #                 train_model = AutoInt(linear_feature_columns,dnn_feature_columns,att_layer_num=6, att_embedding_size=8, att_head_num=6,dnn_hidden_units=(1024, 512,128),seed=16168)
        #                 train_model.load_weights(MODEL_PATH + 'AutoInt_16168_' + y + '.h5')

        #             pre = train_model.predict(test_model_input,batch_size=batch_size * 20)
        #             split_submit5[y] = pre
        #         print(split_submit5.head())
        # #         ########################################AutoInt(3)#############################################
        #         tf.keras.backend.clear_session()
        #         split_submit6 = submit[['userid', 'feedid']]
        #         for y in target:
        #             if (y=='read_comment'or y=='like' or y=='click_avatar'):
        #                 train_model = AutoInt(linear_feature_columns,dnn_feature_columns,att_layer_num=6, att_embedding_size=8, att_head_num=6,dnn_hidden_units=(1024, 512,128),seed=16168)
        #                 train_model.load_weights(MODEL_PATH + 'AutoInt_26273_' + y + '.h5')
        #             elif(y=='click_avatar'):
        #                 train_model = AutoInt(linear_feature_columns,dnn_feature_columns,att_layer_num=6, att_embedding_size=8, att_head_num=8,dnn_hidden_units=(2048, 1024, 512),seed=16168)  # task_dnn_units= (1024, 512, 128), #可能比例越低的目标需要更深的网络来提
        #                 train_model.load_weights(MODEL_PATH + 'AutoInt_26273_' + y + '.h5')
        #             elif(y=='favorite'):
        #                 train_model = AutoInt(linear_feature_columns,dnn_feature_columns,att_layer_num=6, att_embedding_size=8, att_head_num=6,dnn_hidden_units=(2048, 1024, 512),seed=16168)
        #                 train_model.load_weights(MODEL_PATH + 'AutoInt_26273_' + y + '.h5')
        #             else:
        #                 train_model = AutoInt(linear_feature_columns,dnn_feature_columns,att_layer_num=6, att_embedding_size=8, att_head_num=6,dnn_hidden_units=(1024, 512,128),seed=16168)
        #                 train_model.load_weights(MODEL_PATH + 'AutoInt_26273_' + y + '.h5')

        #             pre = train_model.predict(test_model_input,batch_size=batch_size * 20)
        #             split_submit6[y] = pre
        #         print(split_submit6.head())
        # #             ########################################Xdeepfm(1)#############################################
        #         tf.keras.backend.clear_session()
        #         split_submit7 = submit[['userid', 'feedid']]
        #         for y in target:
        #             if (y=='read_comment'or y=='like' or y=='click_avatar'):
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(1024, 512, 128),seed=22231)
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_22231_'+ y +'.h5')
        #             elif(y=='click_avatar'):
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(2048, 1024, 512),seed=22231)
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_22231_'+ y +'.h5')# task_dnn_units=
        #             elif(y=='favorite'):
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(2048, 1024, 512),seed=22231)
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_22231_'+ y +'.h5')# task_dnn_units= (1024, 512, 128), #可能比例越低的目标需要更深的网络来提
        #             else:
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(1024, 512, 128),seed=22231)
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_22231_'+ y +'.h5')
        #             pre = train_model.predict(test_model_input,batch_size=batch_size * 20)
        #             split_submit7[y] = pre
        #         print(split_submit7.head())
        #         #######################################Xdeepfm(2)#############################################
        #         tf.keras.backend.clear_session()
        #         split_submit8 = submit[['userid', 'feedid']]
        #         for y in target:
        #             if (y=='read_comment'or y=='like' or y=='click_avatar'):
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(1024, 512, 128),seed=16168)
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_16168_'+ y +'.h5')
        #             elif(y=='click_avatar'):
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(2048, 1024, 512),seed=16168)
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_16168_'+ y +'.h5')# task_dnn_units=
        #             elif(y=='favorite'):
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(2048, 1024, 512),seed=16168)
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_16168_'+ y +'.h5')# task_dnn_units= (1024, 512, 128), #可能比例越低的目标需要更深的网络来提
        #             else:
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(1024, 512, 128),seed=16168)
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_16168_'+ y +'.h5')
        #             pre = train_model.predict(test_model_input,batch_size=batch_size * 20)
        #             split_submit8[y] = pre
        #         print(split_submit8.head())
        #             ########################################xdeepfm(3)#############################################
        #         tf.keras.backend.clear_session()
        #         split_submit9 = submit[['userid', 'feedid']]
        #         for y in target:
        #             print(y)
        #             if (y=='read_comment'or y=='like' or y=='click_avatar'):
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(2048,1024, 512,128),seed=2021)
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_2021_'+ y + '.h5')
        #             elif(y=='click_avatar'):
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(2048, 1024, 512,128),seed=2021)  # task_dnn_units= (1024, 512, 128), #可能比例越低的目标需要更深的网络来提
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_2021_'+ y + '.h5')
        #             elif(y=='favorite'):
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(2048, 1024, 512,128),seed=2021)  # task_dnn_units= (1024, 512, 128), #可能比例越低的目标需要更深的网络来提
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_2021_'+ y + '.h5')
        #             else:
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(2048,1024, 512,128),seed=2021)
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_2021_'+ y + '.h5')
        #             pre = train_model.predict(test_model_input,batch_size=batch_size * 20)
        #             split_submit9[y] = pre
        #         print(split_submit9.head())
        #          ########################################xdeepfm(4)#############################################
        #         tf.keras.backend.clear_session()
        #         split_submit10 = submit[['userid', 'feedid']]
        #         for y in target:
        #             print(y)
        #             if (y=='read_comment'or y=='like' or y=='click_avatar'):
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(2048, 1024, 512,128),seed=28294)
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_28294_'+ y + '.h5')
        #             elif(y=='click_avatar'):
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(2048, 1024, 512,128),seed=28294)  # task_dnn_units= (1024, 512, 128), #可能比例越低的目标需要更深的网络来提
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_28294_'+ y + '.h5')
        #             elif(y=='favorite'):
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(2048, 1024, 512,128),seed=28294)  # task_dnn_units= (1024, 512, 128), #可能比例越低的目标需要更深的网络来提
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_28294_'+ y + '.h5')
        #             else:
        #                 train_model = xDeepFM(linear_feature_columns,dnn_feature_columns,dnn_hidden_units=(2048, 1024, 512,128),seed=28294)
        #                 train_model.load_weights(MODEL_PATH+'xdeepfm_28294_'+ y + '.h5')
        #             pre = train_model.predict(test_model_input,batch_size=batch_size * 20)
        #             split_submit10[y] = pre
        #         print(split_submit10.head())
        #         yebo_sub = submit[['userid', 'feedid']]
        #         for i in target:
        #             yebo_sub[i] = (split_submit4[i]+split_submit5[i] + split_submit6[i] + split_submit7[i] + split_submit8[i] + split_submit9[i] + split_submit10[i])/7

        ########################################最终所有模型融合#########################################
        # target = ["read_comment", "like", "click_avatar","favorite","forward", "comment", "follow"]
        for i, action in enumerate(target):
            submit[action] = wm_sub[action]  # + 0.2*yebo_sub[action]


    else:  # 测试用，检测model文件是否正常
        tf.keras.backend.clear_session()
        # 4.Define Model,train,predict and evaluate
        train_model = MMOE(dnn_feature_columns, num_tasks=NUM_TASK, expert_dim=28, dnn_hidden_units=(1024, 512, 128),
                           tasks=Tasks,
                           seed=num_repeat * 400)

        train_model.compile(Adagrad(0.01), loss='binary_crossentropy')  # Adagrad

        pred_ans_temp = []
        userid_list = test_data['userid'].astype(str).tolist()
        val_labels = [test_data[y].values for y in target]
        for model_file2 in seed_model_list:
            train_model.load_weights(MODEL_PATH + model_file2)
            val_pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 4)
            pred_ans_temp.append(val_pred_ans)

        # 每个种子取平均
        pred_ans_temp_final = []
        for i, action in enumerate(target):
            for j in range(len(pred_ans_temp)):
                if j == 0:
                    ans_temp = pred_ans_temp[j][i] / len(pred_ans_temp)
                else:
                    ans_temp += pred_ans_temp[j][i] / len(pred_ans_temp)
            pred_ans_temp_final.append(ans_temp)

        uAUC = evaluate_uAUC(val_labels, pred_ans_temp_final, userid_list, target)

    ############################################保存提交文件########################################################
    # 写文件
    file_name = "result.csv"
    if not os.path.exists(SUMIT_DIR):
        os.mkdir(SUMIT_DIR)
    submit_file = os.path.join(SUMIT_DIR, file_name)
    print(submit.head())
    print('Save to: %s' % submit_file)
    submit.to_csv(submit_file, index=False)

    print('Time cost: %.2f s' % (time.time() - t))
    print('all inference finished')

    # 跑完强制结束该进程（必要，deepctr的包调用了thread，因版本问题可能会导致程序无法正常退出）
    os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    tf.app.run(main)
