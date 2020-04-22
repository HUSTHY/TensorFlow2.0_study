"在TensorFlow中完成文本预处理的方案有多种，其中一种比较简单且常用的方案就是使用"
"tf.data.Dataset和.keras.layers.experimental.preprocessing.TextVectorization联合处理"

"一、准备数据"

import os
import sys
sys.path.append(os.getcwd())


import pandas as pd
import numpy as np
import tensorflow as tf
import re,string
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


train_data_path = "data/imdb/train.csv"
test_data_path =  "data/imdb/test.csv"

MAX_WORDS = 10000  # 仅考虑最高频的10000个词
MAX_LEN = 200  # 每个样本保留200个词的长度
BATCH_SIZE = 20


def split_line(line):
    arr = tf.strings.split(line,'\t')
    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]),tf.int32),axis=0)
    text = tf.expand_dims(arr[1],axis=0)
    return (text,label)

ds_train_raw = tf.data.TextLineDataset(filenames=[train_data_path]).map(split_line,num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
ds_test_raw = tf.data.TextLineDataset(filenames=[test_data_path]).map(split_line,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

print('ds_train_raw:',ds_train_raw)
print('ds_test_raw:',ds_test_raw)


def clean_text(text):
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase,'<br/>','')
    cleaned_punctuation = tf.strings.regex_replace(stripped_html,'[%s]' % re.escape(string.punctuation), '')
    return cleaned_punctuation

vectorize_layer = TextVectorization(standardize = clean_text,split = 'whitespace',max_tokens=MAX_WORDS,output_mode='int',output_sequence_length=MAX_LEN)

ds_text = ds_train_raw.map(lambda text,label:text)
vectorize_layer.adapt(ds_text)
print(vectorize_layer.get_vocabulary()[0:10])

ds_train = ds_train_raw.map(lambda text,label:(vectorize_layer(text),label)).prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test_raw.map(lambda text,label:(vectorize_layer(text),label)).prefetch(tf.data.experimental.AUTOTUNE)


if __name__ == '__main__':
    print('tensorflow  study')