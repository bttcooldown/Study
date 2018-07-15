# -*- coding: utf-8 -*-
"""
Created on Mon May  7 20:04:57 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_funtion=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activation_funtion is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_funtion(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5+noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs,1,10,activation_funtion=tf.nn.relu)
prediction = add_layer(l1,10,1,activation_funtion=None)

#计算预测值prediction和真实值的误差，对二者差的平方求和再取平均。
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                    reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#使用变量时，都要对它进行初始化，这是必不可少的。
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#可视化
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
#plt.show()

#下面，让机器开始学习。
for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
#        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.5)


    