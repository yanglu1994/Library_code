from __future__ import division
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import tensorflow as tf
import numpy as np
from math import ceil
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

MOVING_AVERAGE_DECAY = 0.9997
BN_EPSILON = 0.001
BN_DECAY = MOVING_AVERAGE_DECAY
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'
TRAIN = False


class Network(object):
    def __init__(self, train_batch_size, test_batch_size, dropout_rate, base_learning_rate, decay_rate,
                                     optimizeMethod='adam', save_path='segnet_model\default.ckpt'):

        self.optimizeMethod = optimizeMethod
        self.dropout_rate = dropout_rate
        self.base_learning_rate = base_learning_rate
        self.decay_rate = decay_rate

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        # Hyper Parameters
        self.config = []
        self.weights = []
        self.biases = []
        self.data_list = {}

        # Graph Related
        self.tf_train_sample = None
        self.tf_train_labels = None
        self.tf_test_sample = None
        self.tf_test_labels = None

        # Statistic
        self.writer = None
        self.merged_train_summary = None
        self.merged_test_summary = None
        self.merged_test_visualize_summary = None
        self.train_summaries = []
        self.test_summaries = []

        # Save
        self.saver = None
        self.save_path = save_path

    def _get_variable(self, name, shape, initializer, weight_decay=0.0, dtype='float', trainable=True):
        "A little wrapper around tf.get_variable to do weight decay and add to"
        "resnet collection"
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        collections = [tf.GraphKeys.VARIABLES, RESNET_VARIABLES]
        return tf.Variable(name,
                               shape=shape,
                               initializer_value=initializer,
                               dtype=dtype,
                               collections=collections,
                               trainable=trainable)

    def apply_regularization(self, _lambda):
        """
        L2 regularization for the fully connected parameters
        :param _lambda:
        :return:
        """
        regularization = 0.0
        for weight, biases in zip(self.weights, self.biases):
            regularization += tf.nn.l2_loss(weight) + tf.nn.l2_loss(biases)

        return _lambda*regularization

    def define_inputs(self, train_samples_shape, train_labels_shape, test_samples_shape, test_labels_shape):
        with tf.name_scope('inputs'):
            self.tf_train_sample = tf.placeholder(tf.float32, shape=train_samples_shape, name='tf_train_samples')
            self.tf_train_labels = tf.placeholder(tf.float32, shape=train_labels_shape, name='tf_train_labels')
            self.tf_test_sample = tf.placeholder(tf.float32, shape=test_samples_shape, name='tf_test_samples')
            self.tf_test_labels = tf.placeholder(tf.float32, shape=test_labels_shape, name='tf_test_labels')
        return self.tf_train_sample, self.tf_test_sample

    def add_conv(self, data_flow, patch_size, in_depth, out_depth, name, activation=None, train=TRAIN, dropout=False):
        """
        this function dose not define operations in the graph, but only store config in self.conv_layer_config
        :param patch_size:
        :param in_depth:
        :param out_depth:
        :param activation:
        :param pooling:
        :param name:
        :return:
        """
        with tf.name_scope(name + '_model'):
            weights = tf.Variable(
                tf.random_uniform([patch_size, patch_size, in_depth, out_depth],
                                  minval=-tf.sqrt(6/(in_depth+out_depth+1)), maxval=tf.sqrt(6/(in_depth+out_depth+1))),
                # tf.random_normal([patch_size, patch_size, in_depth, out_depth], stddev=2/(in_depth+out_depth+1)),
                name=name+'_weight'
            )
            biases = tf.Variable(
                tf.constant(0.1, shape=[out_depth]), name=name+'_biases'
            )
            base_data = data_flow
            data_flow = tf.nn.conv2d(data_flow, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
            data_flow = data_flow + biases

            if not train:
                self.visualize_filter_map(base_data, name=name+'_base')
                self.visualize_filter_map(data_flow, name=name+'_conv')

            data_flow = self.add_bn(data_flow)

            if activation == 'relu':
                data_flow = tf.nn.relu(data_flow)
                print( name)
                print(data_flow.shape)

            if dropout:
                data_flow = tf.nn.dropout(data_flow, keep_prob=self.dropout_rate)

            self.train_summaries.append(tf.summary.histogram(str(len(self.weights)) + '_weight', weights))
            self.train_summaries.append(tf.summary.histogram(str(len(self.biases)) + '_biases', biases))
        return data_flow

    def add_pool(self, data_flow, pooling_scale, name, train=TRAIN):
        """
        add max_pooling layer config
        :param pooling_scale:
        :return:
        """
        with tf.name_scope(name + '_model'):
            pooling_stride = pooling_scale
            data_flow = tf.nn.max_pool(
                data_flow,
                ksize=[1, pooling_scale, pooling_scale, 1],
                strides=[1, pooling_stride, pooling_stride, 1],
                padding='SAME'
            )
            print(name)
            print(data_flow.shape)
            if not train:
                self.visualize_filter_map(data_flow, name=name + '_pooling')
        return data_flow

    def add_us1(self, data_flow, patch_size, in_channels, out_channels, output_shape, name, activation='relu', train=TRAIN):
        """
        add up_sampling layer to self.us_layer_config
        :return:
        """
        with tf.name_scope(name + '_model'):
            weight = tf.Variable(
                tf.random_uniform([patch_size, patch_size, out_channels, in_channels],
                                  minval=-tf.sqrt(6 / (in_channels + out_channels + 1)),
                                  maxval=tf.sqrt(6 / (in_channels + out_channels + 1))),
                # tf.random_normal([patch_size, patch_size, out_channels, in_channels], stddev=2/(in_channels+out_channels+1)),
                name=name+'_weight'
            )
            biases = tf.Variable(
                tf.constant(0.1, shape=[out_channels])
            )
            data_flow = tf.nn.conv2d_transpose(data_flow, filter=weight, output_shape=output_shape,
                                               strides=[1, 2, 2, 1], padding='SAME')
            data_flow = data_flow + biases
            if not train:
                self.visualize_filter_map(data_flow, name=name + '_us')
            if activation == 'relu':
                data_flow = tf.nn.relu(data_flow)
                print (name)
                print (data_flow.shape)
                if not train:
                    self.visualize_filter_map(data_flow, name=name + '_relu')
            else:
                raise Exception('Activation func error')
        return data_flow

    def get_deconv_filter(self, f_shape):
        """
          reference: https://github.com/MarvinTeichmann/tensorflow-fcn
        """
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)

    def add_us(self, data_flow, patch_size, in_channels, out_channels, output_shape, name,activation='relu', train=TRAIN):
        # output_shape = [b, w, h, c]
        # sess_temp = tf.InteractiveSession()
        f_shape = [patch_size, patch_size, out_channels, in_channels]
        with tf.variable_scope(name + '_model'):
            weights = self.get_deconv_filter(f_shape)
            biases = tf.Variable(
                            tf.constant(0.1, shape=[out_channels])
                        )
            data_flow = tf.nn.conv2d_transpose(data_flow, weights, output_shape,
                                            strides=[1, 2, 2, 1], padding='SAME')
            data_flow = data_flow + biases
            if not train:
                self.visualize_filter_map(data_flow, name=name + '_us')
            if activation == 'relu':
                data_flow = tf.nn.relu(data_flow)
                print(name)
                print(data_flow.shape)
                if not train:
                    self.visualize_filter_map(data_flow, name=name + '_relu')
            else:
                raise Exception('Activation func error')
        return data_flow

    def unpool_with_argmax(self, dataflow, ind, name=None, ksize=[1, 2, 2, 1]):

        """
           Unpooling layer after max_pool_with_argmax.
           Args:
               pool:   max pooled output tensor
               ind:      argmax indices
               ksize:     ksize is the same as for the pool
           Return:
               unpool:    unpooling tensor
        """
        with tf.variable_scope(name):
            input_shape = dataflow.get_shape().as_list()
            output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

            flat_input_size = np.prod(input_shape) # np.prob multiply one by one
            flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

            dataflow = tf.reshape(dataflow, [flat_input_size])
            batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
            b = tf.ones_like(ind) * batch_range
            b = tf.reshape(b, [flat_input_size, 1])
            ind_ = tf.reshape(ind, [flat_input_size, 1])
            ind_ = tf.concat([b, ind_], 1)

            dataflow = tf.scatter_nd(ind_, dataflow, shape=flat_output_shape)
            dataflow = tf.reshape(dataflow, output_shape)
            return dataflow

    def add_concat(self, value, name):
        with tf.name_scope(name + '_model'):
            data_flow = tf.concat(value, axis=3)
            print (name)
            print (data_flow.shape)
        return data_flow

    def add_bn(self, data_flow, train=TRAIN):
        train = tf.convert_to_tensor(train, dtype='bool', name='train')
        x_shape = data_flow.get_shape()
        params_shape = x_shape[-1:]

        axis = list(range(len(x_shape) - 1))
        beta = tf.Variable(tf.zeros(shape=params_shape), name='beta')
        gamma = tf.Variable(tf.ones(shape=params_shape), name='gamma')
        # beta = self._get_variable('beta', params_shape, initializer=tf.zeros_initializer)
        # gamma = self._get_variable('gamma', params_shape, initializer=tf.ones_initializer)
        mean, variance = tf.nn.moments(data_flow, axis)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([mean, variance])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(mean), tf.identity(variance)

        _mean, _variance = control_flow_ops.cond(train, mean_var_with_update, lambda:(ema.average(mean), ema.average(variance)))
        data_flow = tf.nn.batch_normalization(data_flow, _mean, _variance, beta, gamma, BN_EPSILON)

        return data_flow

    def define_model(self, logits, train=TRAIN):
        # Training computation
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_train_labels,
                                                                 logits=tf.clip_by_value(logits, 1e-8, 1.0)))
            self.loss += self.apply_regularization(_lambda=0.05)
            self.train_summaries.append(tf.summary.scalar('Loss', self.loss))

        # Learning rate decay
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            learning_rate=self.base_learning_rate,
            global_step=global_step*self.train_batch_size,
            decay_steps=100,
            decay_rate=self.decay_rate,
            staircase=True

        )

        # Optimizer
        with tf.name_scope('optimizer'):
            if self.optimizeMethod == 'gradient':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
            elif self.optimizeMethod == 'mometum':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(self.loss)
            elif self.optimizeMethod == 'adam':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # Predictions for the training, validation and test dataset
        prediction = tf.cast(tf.argmax(logits, axis=3), tf.float32)
        prediction = tf.expand_dims(input=prediction, axis=3)
        # shape = prediction.get_shape().as_list()
        # shape.append(1)
        # prediction = tf.reshape(prediction, shape)
        with tf.name_scope('train'):
            self.train_prediction = logits
            tf.add_to_collection('prediction', self.train_prediction)
            self.train_summaries.append(tf.summary.image(name='prediction', tensor=prediction))

        with tf.name_scope('test'):
            self.test_prediction = logits
            self.test_summaries.append(tf.summary.image(name='prediction', tensor=prediction))

        self.saver = tf.train.Saver(tf.global_variables())
        if train:
            self.merged_train_summary = tf.summary.merge(self.train_summaries)
        else:
            self.merged_test_summary = tf.summary.merge(self.test_summaries)

    def train(self, train_samples, train_labels, data_iterator, iteration_steps):
        self.writer = tf.summary.FileWriter('board', tf.get_default_graph())
        with tf.Session(graph=tf.get_default_graph()) as session:
            tf.global_variables_initializer().run()

            print('Start Training')
            # batch 1000
            for i, samples, labels in data_iterator(train_samples, train_labels, iteration_steps=iteration_steps,
                                                    chunkSize=self.train_batch_size):
                # _, l, predictions, summary = session.run(
                #     [self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
                #     feed_dict={self.tf_train_sample: samples, self.tf_train_labels: labels}
                # )
                _, l, predictions, summary = session.run(
                    [self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
                    feed_dict={self.tf_train_sample: samples, self.tf_train_labels: labels}
                )
                self.writer.add_summary(summary, i)
                # labels is True Labels
                accuracy = self.accuracy(predictions, labels, i)
                if i % 1 == 0:
                    print('Minibatch loss at step %d: %f' % (i, l))
                    print('Minibatch accuracy: %.1f%%' % accuracy)
            # ###

            import os
            if os.path.isdir(self.save_path.split('\ ')[0]):
                save_path = self.saver.save(session, self.save_path)
                print("Model saved in file: %s" % save_path)
            else:
                os.makedirs(self.save_path.split('\ ')[0])
                save_path = self.saver.save(session, self.save_path)
                print("Model saved in file: %s" % save_path)

    def test(self, test_samples, test_labels, data_iterator):
        if self.saver is None:
            self.saver = tf.train.Saver()
        if self.writer is None:
            self.writer = tf.summary.FileWriter('segnet_board', tf.get_default_graph())

        print('Before session')
        with tf.Session(graph=tf.get_default_graph()) as session:
            self.saver.restore(session, self.save_path)
            accuracies = []
            confusionMatrices = []
            for i, samples, labels in data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):

                result, summary = session.run(
                    [self.test_prediction, self.merged_test_summary],
                    feed_dict={self.tf_test_sample: samples, self.tf_test_labels: labels}
                )
                self.writer.add_summary(summary, i)
                accuracy= self.accuracy(result, labels, i, need_confusion_matrix=True)
                accuracies.append(accuracy)
                # confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            # self.print_confusion_matrix(np.add.reduce(confusionMatrices))

    def accuracy(self, predictions, labels,i, train=TRAIN, need_confusion_matrix=False):
        predictions = np.argmax(predictions, 3)
        labels = np.argmax(labels, 3)
        shape = list(predictions.shape)
        accuracy = (100.0 * np.sum(predictions == labels) / (shape[0]*shape[1] *shape[2]))
        shape = shape.append(1)
        predictions = np.reshape(predictions, shape)
        if not train:
            path = 'predictions\predict_%s' % i
            np.save(path, predictions)
        # if i == 40:
        #     image1 = np.reshape(predictions[-1], [400, 400])
        #     image2 = np.reshape(labels[-1], [400, 400])
        #     fig = plt.figure()
        #     ax = fig.add_subplot(121)
        #     ax.imshow(image1)
        #     ax = fig.add_subplot(122)
        #     ax.imshow(image2)
        #     plt.show()
        return accuracy

    def visualize_filter_map(self, tensor, name):
        filter_map = tensor[-1]
        filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
        shape = filter_map.get_shape().as_list()
        shape.append(1)
        filter_map = tf.reshape(filter_map, shape)
        self.test_summaries.append(tf.summary.image(name, tensor=filter_map, max_outputs=shape[0]))

    def print_confusion_matrix(self, confusionMatrix):
        print('Confusion    Matrix:')
        for i, line in enumerate(confusionMatrix):
            print(line, line[i] / np.sum(line))
        a = 0
        for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
            a += (column[i] / np.sum(column)) * (np.sum(column) / 26000)
            print(column[i] / np.sum(column),)
        print('\n', np.sum(confusionMatrix), a)











