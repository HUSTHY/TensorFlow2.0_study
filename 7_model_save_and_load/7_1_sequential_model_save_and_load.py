"""
sequential模型可以保存为完整的.h5模型和savedModel——(这是一个文件夹）
"""

import tensorflow as tf
from tensorflow.keras import models,layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
def auto_distribute_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu,True)
        except RuntimeError as e:
            print(e)

def create_model():
    model = models.Sequential()
    model.add(layers.Embedding(200,768,input_length=512))
    model.add(layers.Conv1D(filters=64,kernel_size=5,activation='relu'))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation="softmax"))
    return model

if __name__ == '__main__':
    auto_distribute_gpu_memory()
    seqential_model = create_model()
    seqential_model.compile(optimizer='Nadam',
            loss='binary_crossentropy',
            metrics=['accuracy',"AUC"])
    seqential_model.summary()
    seqential_model.save('seqential_model.h5')
    seqential_model.save('SavedModel',save_format='tf')

    #保存完整的模型结构和参数
    h5_model = models.load_model('seqential_model.h5')
    h5_model.summary()

    savedModel = models.load_model('SavedModel')
    savedModel.summary()
