"""
数据管道Dataset
一般使用 tf.data.Dataset.from_tensor_slices
当然还有其他的 tf.data.Dataset.from_generator
通过tfrecords文件方式构建数据管道较为复杂
需要对样本构建tf.Example后压缩成字符串写到tfrecords文件
读取后再解析成tf.Example
"""

# 1、使用from_tensor_slices
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='1'

import tensorflow as tf
from sklearn import datasets

iris = datasets.load_iris()
ds1 = tf.data.Dataset.from_tensor_slices((iris['data'],iris['target']))
ds1 = ds1.batch(5).prefetch(buffer_size= tf.data.experimental.AUTOTUNE) #给出batch_size;有点类似torch.DataLoader
for features,labels in ds1:
    tf.print(features)


# # 2、使用from_generator----从Python generator构建数据管道
# from tensorflow.keras.preprocessing.image import  ImageDataGenerator
# image_generator = ImageDataGenerator(rescale=1.0/255).flow_from_directory('../data/cifar2/test/',target_size=(32,32),batch_size=20,class_mode='binary')
# def generator():
#     for features,labels in image_generator:
#         yield (features,labels)
# ds2 = tf.data.Dataset.from_generator(generator,output_types=(tf.float32,tf.int32))
# for features,labels in ds2:
#     tf.print(features,labels)

# 3、从csv文件中构建
ds3 = tf.data.experimental.make_csv_dataset(
file_pattern = ["../data/titanic/train.csv","../data/titanic/test.csv"],
      batch_size=5,
      label_name="Survived",
      na_value="",
      num_epochs=1,
      ignore_errors=True
)
for data,label in ds3.take(2):
    tf.print(data,label)

# 4、从文本文件构建数据通道——tf.data.TextLineDataset
ds4 = tf.data.TextLineDataset(
    filenames=["../data/titanic/train.csv","../data/titanic/test.csv"]
).skip(1)
for line in ds4.take(10):
    tf.print(line)