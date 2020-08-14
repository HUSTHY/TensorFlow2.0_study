"""
Tensorflow的张量和numpy中的array很类似
tensor有2中类型一种是常量constant、一种是变量variable
constant常量的值在计算图中不可以被重新赋值，variable变量可以在计算图中用assign等算子重新赋值

"""

import tensorflow as tf

if __name__ == '__main__':

    """constant——常量"""

    a = tf.constant(1)
    b = tf.constant(1,dtype=tf.int64)
    c = tf.constant(3.14)
    d = tf.constant(3.14,dtype=tf.float64)
    e = tf.constant(3.14,dtype=tf.double)
    f = tf.constant('Tensorflow')
    g = tf.constant(True)
    print(type(a), a)
    print(type(b), b)
    print(type(c), c)
    print(type(d), d)
    print(type(e), e)
    print(type(f), f)
    print(type(g), g)

    """张量to变量"""
    c = tf.constant([1.0,2.0])
    print(c)
    print(id(c))
    c += tf.constant([0.5,0.4])
    print(c)
    print(id(c))


    v = tf.Variable([1.0,2.0])
    print(v)
    print(id(v))
    # v.assign_sub()做减法
    v.assign_add([1.0,0.6])
    print(v)
    print(id(v))