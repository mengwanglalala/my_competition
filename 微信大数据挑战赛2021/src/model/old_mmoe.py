# -*- coding:utf-8 -*-

import tensorflow as tf

from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.utils import combined_dnn_input
from deepctr.layers.core import PredictionLayer, DNN

from tensorflow.python.keras.initializers import glorot_normal
from tensorflow.python.keras.layers import Layer
# class Self_Attention(Layer):
#
#     def __init__(self, output_dim, **kwargs):
#         self.output_dim = output_dim
#         super(Self_Attention, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # 为该层创建一个可训练的权重
#         # inputs.shape = (batch_size, time_steps, seq_len)
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(3, input_shape[2], self.output_dim),
#                                       initializer='uniform',
#                                       trainable=True)
#
#         super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它
#
#     def call(self, x):
#         WQ = K.dot(x, self.kernel[0])
#         WK = K.dot(x, self.kernel[1])
#         WV = K.dot(x, self.kernel[2])
#
#         print("WQ.shape", WQ.shape)
#
#         print("K.permute_dimensions(WK, [0, 2, 1]).shape", K.permute_dimensions(WK, [0, 2, 1]).shape)
#
#         QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
#
#         QK = QK / (64 ** 0.5)
#
#         QK = K.softmax(QK)
#
#         print("QK.shape", QK.shape)
#
#         V = K.batch_dot(QK, WV)
#
#         return V
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[1], self.output_dim)
class MMoELayer(Layer):
    """
    The Multi-gate Mixture-of-Experts layer in MMOE model
      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.
      Output shape
        - A list with **num_tasks** elements, which is a 2D tensor with shape: ``(batch_size, units_experts)`` .
      Arguments
        - **num_tasks**: integer, the number of tasks, equal to the number of outputs.
        - **num_experts**: integer, the number of experts.
        - **units_experts**: integer, the dimension of each output of MMOELayer.
    References
      - [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
    """

    def __init__(self, units_experts, num_experts, num_tasks,
                 use_expert_bias=True, use_gate_bias=True, expert_activation='relu', gate_activation='softmax',
                 expert_bias_initializer='zeros', gate_bias_initializer='zeros', expert_bias_regularizer=None,
                 gate_bias_regularizer=None, expert_bias_constraint=None, gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling', gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None, gate_kernel_regularizer=None, expert_kernel_constraint=None,
                 gate_kernel_constraint=None, activity_regularizer=None, **kwargs):
        super(MMoELayer, self).__init__(**kwargs)

        self.units_experts = units_experts
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = tf.keras.initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = tf.keras.initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = tf.keras.regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = tf.keras.regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = tf.keras.constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = tf.keras.constraints.get(gate_kernel_constraint)

        # Activation parameter
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = tf.keras.initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = tf.keras.initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = tf.keras.regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = tf.keras.regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = tf.keras.constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = tf.keras.constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []

        for i in range(self.num_experts):
            self.expert_layers.append(tf.keras.layers.Dense(self.units_experts, activation=self.expert_activation,
                                                            use_bias=self.use_expert_bias,
                                                            kernel_initializer=self.expert_kernel_initializer,
                                                            bias_initializer=self.expert_bias_initializer,
                                                            kernel_regularizer=self.expert_kernel_regularizer,
                                                            bias_regularizer=self.expert_bias_regularizer,
                                                            activity_regularizer=self.activity_regularizer,
                                                            kernel_constraint=self.expert_kernel_constraint,
                                                            bias_constraint=self.expert_bias_constraint,
                                                            name='expert_net_{}'.format(i)))
        for i in range(self.num_tasks):
            self.gate_layers.append(tf.keras.layers.Dense(self.num_experts, activation=self.gate_activation,
                                                          use_bias=self.use_gate_bias,
                                                          kernel_initializer=self.gate_kernel_initializer,
                                                          bias_initializer=self.gate_bias_initializer,
                                                          kernel_regularizer=self.gate_kernel_regularizer,
                                                          bias_regularizer=self.gate_bias_regularizer,
                                                          activity_regularizer=self.activity_regularizer,
                                                          kernel_constraint=self.gate_kernel_constraint,
                                                          bias_constraint=self.gate_bias_constraint,
                                                          name='gate_net_{}'.format(i)))

    def call(self, inputs, **kwargs):

        expert_outputs, gate_outputs, final_outputs = [], [], []

        # inputs: (batch_size, embedding_size)
        for expert_layer in self.expert_layers:
            expert_output = tf.expand_dims(expert_layer(inputs), axis=2)
            expert_outputs.append(expert_output)

        # batch_size * units * num_experts
        expert_outputs = tf.concat(expert_outputs, 2)

        # [(batch_size, num_experts), ......]
        for gate_layer in self.gate_layers:
            # O_seq = Self_Attention(128)(gate_layer)
            # O_seq = GlobalAveragePooling1D()(O_seq)
            # O_seq = Dropout(0.5)(O_seq)
            #
            # outputs = Dense(self.num_experts, activation='softmax')(O_seq)
            #
            # gate_outputs.append(outputs)

            gate_outputs.append(gate_layer(inputs))

        for gate_output in gate_outputs:
            # (batch_size, 1, num_experts)
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)

            # (batch_size * units * num_experts) * (batch_size, 1 * units, num_experts)
            weighted_expert_output = expert_outputs * tf.keras.backend.repeat_elements(expanded_gate_output,
                                                                                       self.units_experts, axis=1)

            # (batch_size, units)
            final_outputs.append(tf.reduce_sum(weighted_expert_output, axis=2))

        # [(batch_size, units), ......]   size: num_task
        return final_outputs

    def get_config(self, ):
        config = {'units_experts': self.units_experts, 'num_experts': self.num_experts, 'num_tasks': self.num_tasks}
        base_config = super(MMoELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def MMOE(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
         l2_reg_embedding=1e-5  , l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):#5e-6 1e-5

    features = build_input_features(dnn_feature_columns)

    inputs_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    # linear

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                  False, seed=seed)(dnn_input)
    mmoe_outs = MMoELayer(units_experts=expert_dim, num_tasks=num_tasks, num_experts=num_experts,
                            name='mmoe_layer')(dnn_out)


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
