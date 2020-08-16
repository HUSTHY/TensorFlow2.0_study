"""
监督学习的目标函数由损失函数和正则化项组成。（Objective = Loss + Regularization）
在keras中可以使用kernel_regularizer和bias_regularizer来指定L1还是L2正则化
kernel_constraint和bias_constraint来约束权重取值
mean_squared_error 回归模型通常使用均方损失函数
binary_crossentropy  二分类模型、二元交叉熵
categorical_crossentropy 类别交叉熵——多分类模型、label是one-hot
sparse_categorical_crossentropy 稀疏类别交叉熵——多分类模型、label不是one-hot
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import  tensorflow as tf
from tensorflow.keras import layers,models,regularizers,constraints
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01),
                kernel_constraint = constraints.MaxNorm(max_value=2, axis=0)))
model.add(layers.Dense(10,
        kernel_regularizer=regularizers.l1_l2(0.01,0.01),activation = "sigmoid"))
model.compile(optimizer = "rmsprop",
        loss = "binary_crossentropy",metrics = ["AUC"])
model.summary()