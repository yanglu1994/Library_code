# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
print (mnist.train.images.shape)

lr = 1e-3
batch_size = 32
# 每个时刻的输入特征是28维，每个时刻输入一行，一行有28个像素
input_size = 28
# 时序持续长度是28，每做一次预测，需要先输入28行
timestep_size = 28
# 每个隐含层的结点数
hidden_size = 256
# LSTM_layer层数
layer_num = 1
class_num = 10
training_iters = 10000

_x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32, [])
X = tf.reshape(_x, [-1, 28, 28])

# 对weight biases 初始值的定义
weights = {
    'in': tf.Variable(tf.random_normal((input_size, hidden_size))),
    'out': tf.Variable(tf.random_normal((hidden_size, class_num)))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[hidden_size, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[class_num, ]))
}


def RNN(X, weights, biases):
    X = tf.reshape(X, [-1, input_size])
    X_in = tf.matmul(X, weights['in']) +biases['in']
    X_in = tf.reshape(X_in,[-1, timestep_size, hidden_size])
    lstm_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']+biases['out'])
    return results


def train():
    pred = RNN(X, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        step = 0
        while step*batch_size < training_iters:
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run([train_op], feed_dict={
                _x: batch_xs,
                y: batch_ys,
            })
            if step % 20 == 0:
                print(sess.run(accuracy, feed_dict={
                    _x: batch_xs,
                    y: batch_ys,
                }))
            step += 1



def model():
    # 定义一层LSTM_cell
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)

    # 添加dropout layer, 设置output_keep_prob
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)

    # 调用MultiRnnCell 实现多层LSTM
    mlstm_cell = rnn.MultiRNNCell([lstm_cell]*layer_num, state_is_tuple=True)

    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

    # outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state,time_major=False)
    # print(outputs.shape)
    # h_state = outputs[:, -1, :]

    outputs = []
    state = init_state
    with tf.variable_scope('RNN'):
        for timestep in range(timestep_size):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            single_input = X[:, timestep, :]
            (cell_output, state) = mlstm_cell(single_input, state)
            outputs.append(cell_output)
    h_state = outputs[-1]

    W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
    y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)

    cross_entropy = -tf.reduce_mean(y * tf.log(y_pre))
    train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            _batch_size = 128
            batch = mnist.train.next_batch(_batch_size)
            if (i + 1) % 200 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={
                    _x: batch[0], y: batch[1], keep_prob: 1.0, batch_size: _batch_size})
                # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
                print("Iter%d, step %d, training accuracy %g" % (mnist.train.epochs_completed, (i + 1), train_accuracy))
            sess.run(train_op, feed_dict={_x: batch[0], y: batch[1], keep_prob: 0.5, batch_size: _batch_size})

        # 计算测试数据的准确率
        print("test accuracy %g" % sess.run(accuracy, feed_dict={
            _x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0, batch_size: mnist.test.images.shape[0]}))

if __name__=="__main__":
    train()