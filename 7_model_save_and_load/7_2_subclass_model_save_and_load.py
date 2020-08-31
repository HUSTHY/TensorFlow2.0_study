"""
sequential模型可以保存为完整的.h5模型和savedModel——(这是一个文件夹）
"""

import tensorflow as tf
from tensorflow.keras import models,layers
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
def auto_distribute_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu,True)
        except RuntimeError as e:
            print(e)

class Mymodel(models.Model):
    def __init__(self):
        super(Mymodel,self).__init__()
        self.emd = layers.Embedding(200,768)
        self.conv = layers.Conv1D(filters=64,kernel_size=5,activation='relu')
        self.pool = layers.MaxPool1D(2)
        self.flat = layers.Flatten()
        self.cl = layers.Dense(2, activation="softmax")


    def call(self, inputs, training=None, mask=None):
        x = self.emd(inputs)
        x = self.conv(x)
        x = self.pool(x)
        x = self.flat(x)
        output = self.cl(x)
        return output

if __name__ == '__main__':
    auto_distribute_gpu_memory()
    sub_class_model = Mymodel()
    sub_class_model.compile(optimizer='Nadam',
            loss='binary_crossentropy',
            metrics=['accuracy',"AUC"])
    #没有build的话，summary()的调用也会出错的
    sub_class_model.build(input_shape=(None,512))
    sub_class_model.summary()

    #TensorSpec创建一个“无实际数据的张量”，指定它的形状，作为模型的输入
    shape = tf.TensorSpec(shape=(100, 512), dtype=tf.dtypes.int32, name=None)
    #没有_set_inputs这一步模型保存会报错
    sub_class_model._set_inputs(shape)
    sub_class_model.save('sub_class_model',save_format='tf')

    #保存完整的模型结构和参数
    sub_class_h5_model = models.load_model('sub_class_model')
    sub_class_h5_model.summary()
