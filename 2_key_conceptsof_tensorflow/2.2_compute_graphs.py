"""
计算图是TensorFlow中的一个概念，模型定义和参数求解方式抽象之后，生成一个
计算逻辑，用图表示就是一个计算图
根据TensorFlow版本的不同，这里有3中计算图
1.0时代就是静态计算图，这个效率比较高，但是学习起来比较麻烦
2.0时代就出现了动态计算图——没使用一个算子后，该算子就会被动态的加入到隐含的默认计算图中国计算
缺点是运行效率很低，有点就是和python代码一样简单、易理解
运行效率低的原因就是动态图会多次python进程和TensorFlow的C++进程之间的通信

Autograph——更加现进的方法就是使用@tf.function装饰器把python函数转化为TensorFlow的计算图，不用手动开启session代码
"""
import tensorflow as tf

@tf.function
def strjoin(x,y):
    z = tf.strings.join([x,y],separator='_')
    tf.print(z)
    return z

result = strjoin(tf.constant('hello'),tf.constant('TensorFlow'))
print(result)