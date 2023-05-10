import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from deepctr.models import xDeepFM, DeepFM, DCN
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from tensorflow.python.keras.optimizers import Adam

from deepctr.feature_column import input_from_feature_columns, get_linear_logit,build_input_features
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.interaction import FM
from deepctr.layers.utils import concat_func, combined_dnn_input
from .mmoe import MMoE
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils import multi_gpu_model

import tensorflow as tf
from tensorflow.python.keras.layers import   Input
import os
import tensorflow as tf
from tensorflow.python.keras.layers import (Concatenate, Dense, Permute, multiply)

from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, build_input_features
from deepctr.inputs import get_varlen_pooling_list, create_embedding_matrix, embedding_lookup, varlen_embedding_lookup, \
    get_dense_input
from deepctr.layers.core import DNN, PredictionLayer
from deepctr.layers.sequence import AttentionSequencePoolingLayer, DynamicGRU
from deepctr.layers.utils import concat_func, reduce_mean, combined_dnn_input


def auxiliary_loss(h_states, click_seq, noclick_seq, mask, stag=None):
    #:param h_states:
    #:param click_seq:
    #:param noclick_seq: #[B,T-1,E]
    #:param mask:#[B,1]
    #:param stag:
    #:return:
    hist_len, _ = click_seq.get_shape().as_list()[1:]
    mask = tf.sequence_mask(mask, hist_len)
    mask = mask[:, 0, :]

    mask = tf.cast(mask, tf.float32)

    click_input_ = tf.concat([h_states, click_seq], -1)

    noclick_input_ = tf.concat([h_states, noclick_seq], -1)

    auxiliary_nn = DNN([100, 50, 1], activation='sigmoid')

    click_prop_ = auxiliary_nn(click_input_, stag=stag)[:, :, 0]

    noclick_prop_ = auxiliary_nn(noclick_input_, stag=stag)[
                    :, :, 0]  # [B,T-1]

    try:
        click_loss_ = - tf.reshape(tf.log(click_prop_),
                                   [-1, tf.shape(click_seq)[1]]) * mask
    except:
        click_loss_ = - tf.reshape(tf.compat.v1.log(click_prop_),
                                   [-1, tf.shape(click_seq)[1]]) * mask
    try:
        noclick_loss_ = - \
                            tf.reshape(tf.log(1.0 - noclick_prop_),
                                       [-1, tf.shape(noclick_seq)[1]]) * mask
    except:
        noclick_loss_ = - \
                            tf.reshape(tf.compat.v1.log(1.0 - noclick_prop_),
                                       [-1, tf.shape(noclick_seq)[1]]) * mask

    loss_ = reduce_mean(click_loss_ + noclick_loss_)

    return loss_

def interest_evolution(concat_behavior, deep_input_item, user_behavior_length, gru_type="GRU", use_neg=False,
                       neg_concat_behavior=None, att_hidden_size=(64, 16), att_activation='sigmoid',
                       att_weight_normalization=False, ):
    if gru_type not in ["GRU", "AIGRU", "AGRU", "AUGRU"]:
        raise ValueError("gru_type error ")
    aux_loss_1 = None
    embedding_size = None
    rnn_outputs = DynamicGRU(embedding_size, return_sequence=True,
                             name="gru1")([concat_behavior, user_behavior_length])

    if gru_type == "AUGRU" and use_neg:
        aux_loss_1 = auxiliary_loss(rnn_outputs[:, :-1, :], concat_behavior[:, 1:, :],

                                    neg_concat_behavior[:, 1:, :],

                                    tf.subtract(user_behavior_length, 1), stag="gru")  # [:, 1:]

    if gru_type == "GRU":
        rnn_outputs2 = DynamicGRU(embedding_size, return_sequence=True,
                                  name="gru2")([rnn_outputs, user_behavior_length])
        # attention_score = AttentionSequencePoolingLayer(hidden_size=att_hidden_size, activation=att_activation, weight_normalization=att_weight_normalization, return_score=True)([
        #     deep_input_item, rnn_outputs2, user_behavior_length])
        # outputs = Lambda(lambda x: tf.matmul(x[0], x[1]))(
        #     [attention_score, rnn_outputs2])
        # hist = outputs
        hist = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size, att_activation=att_activation,
                                             weight_normalization=att_weight_normalization, return_score=False)([
            deep_input_item, rnn_outputs2, user_behavior_length])

    else:  # AIGRU AGRU AUGRU

        scores = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size, att_activation=att_activation,
                                               weight_normalization=att_weight_normalization, return_score=True)([
            deep_input_item, rnn_outputs, user_behavior_length])

        if gru_type == "AIGRU":
            hist = multiply([rnn_outputs, Permute([2, 1])(scores)])
            final_state2 = DynamicGRU(embedding_size, gru_type="GRU", return_sequence=False, name='gru2')(
                [hist, user_behavior_length])
        else:  # AGRU AUGRU
            final_state2 = DynamicGRU(embedding_size, gru_type=gru_type, return_sequence=False,
                                      name='gru2')([rnn_outputs, user_behavior_length, Permute([2, 1])(scores)])
        hist = final_state2
    return hist, aux_loss_1




def MMOE(dnn_feature_columns, history_feature_list, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
            gru_type="AUGRU", use_negsampling=False, alpha=1.0, use_bn=False,
         att_hidden_units=(64, 16), att_activation="dice", att_weight_normalization=True,
         l2_reg_embedding=1e-5  , l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):#5e-6 1e-5
    """Instantiates the Multi-gate Mixture-of-Experts architecture.
    可以改进，比如每个输出目标的权重，不平衡的dnn层数
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param num_tasks: integer, number of tasks, equal to number of outputs, must be greater than 1.
    :param tasks: list of str, indicating the loss of each tasks, ``"binary"`` for  binary logloss, ``"regression"`` for regression loss. e.g. ['binary', 'regression']
    :param num_experts: integer, number of experts.
    :param expert_dim: integer, the hidden units of each expert.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of shared-bottom DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param task_dnn_units: list,list of positive integer or empty list, the layer number and units in each layer of task-specific DNN
    :param seed: integer ,to use as random seed.
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN

    :return: a Keras model instance
    """
    print('baseline')
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    # features = build_input_features(dnn_feature_columns)
    #
    # inputs_list = list(features.values())
    #
    # sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
    #                                                                      l2_reg_embedding, seed)

    # linear

    features = build_input_features(dnn_feature_columns)

    user_behavior_length = features["feedid_len"]

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    history_feature_columns = []
    neg_history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))
    print(history_feature_list)
    neg_history_fc_names = list(map(lambda x: "neg_" + x, history_fc_names))
    print(neg_history_fc_names)
    print(history_fc_names)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        elif feature_name in neg_history_fc_names:
            neg_history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    inputs_list = list(features.values())

    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, seed, prefix="",
                                             seq_mask_zero=False)

    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                      return_feat_list=history_feature_list, to_list=True)

    keys_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns,
                                     return_feat_list=history_fc_names, to_list=True)
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns,
                                          mask_feat_list=history_feature_list, to_list=True)
    dense_value_list = get_dense_input(features, dense_feature_columns)

    sequence_embed_dict = varlen_embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns,
                                                  to_list=True)
    dnn_input_emb_list += sequence_embed_list
    keys_emb = concat_func(keys_emb_list)
    deep_input_emb = concat_func(dnn_input_emb_list)
    query_emb = concat_func(query_emb_list)
    print(embedding_dict)
    print(features)
    print(neg_history_feature_columns)
    print(neg_history_fc_names)
    if use_negsampling:

        neg_uiseq_embed_list = embedding_lookup(embedding_dict, features, neg_history_feature_columns,
                                                neg_history_fc_names, to_list=True)

        neg_concat_behavior = concat_func(neg_uiseq_embed_list)

    else:
        neg_concat_behavior = None
    hist, aux_loss_1 = interest_evolution(keys_emb, query_emb, user_behavior_length, gru_type=gru_type,
                                          use_neg=use_negsampling, neg_concat_behavior=neg_concat_behavior,
                                          att_hidden_size=att_hidden_units,
                                          att_activation=att_activation,
                                          att_weight_normalization=att_weight_normalization, )

    deep_input_emb = Concatenate()([deep_input_emb, hist])

    deep_input_emb = tf.keras.layers.Flatten()(deep_input_emb)

    dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed=seed)(dnn_input)
    # mmoe_outs = MMOELayer(units_experts=expert_dim, num_tasks=num_tasks, num_experts=num_experts,
    #                         name='mmoe_layer')(dnn_out)

    mmoe_outs = MMoE(units=expert_dim, num_experts=num_experts, num_tasks=num_tasks)(dnn_out)


    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units[i], dnn_activation, l2_reg_dnn, 0, False, seed=seed)(mmoe_out) for i, mmoe_out in
                     enumerate(mmoe_outs)]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(
            1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list,
                                  outputs=task_outputs)

    if use_negsampling:
        model.add_loss(alpha * aux_loss_1)
    try:
        tf.keras.backend.get_session().run(tf.global_variables_initializer())
    except:
        tf.compat.v1.keras.backend.get_session().run(tf.compat.v1.global_variables_initializer())
        tf.compat.v1.experimental.output_all_intermediates(True)
    return model
