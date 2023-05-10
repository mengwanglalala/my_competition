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




def MMOE(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
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

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    # linear

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
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
    return model
