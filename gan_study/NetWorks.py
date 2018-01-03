import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.gan as tfgan


def _generator_helper(noise, is_conditional, one_hot_labels, weight_decay):
    with tf.contrib.framework.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(noise, 1024)
        if is_conditional:
            net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
        net = layers.fully_connected(noise, 100*100*128)
        net = tf.reshape(net, [-1, 100, 100, 128])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        net = layers.conv2d(net, 1, [4, 4], normalizer_fn=None, activation_fn=tf.tanh)

        return net


def unconditional_generator(noise, weight_decay=2.5e-5):
    return _generator_helper(noise, False, None, weight_decay)


def conditonal_generator(inputs, weight_decay=2.5e-5):
    noise, one_hot_labels = inputs
    return _generator_helper(noise, True, one_hot_labels, weight_decay)


_leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)


def _discriminator_helper(img, is_conditional, one_hot_labels, weight_decay):
    with tf.contrib.framework.arg_scope(
        [layers.conv2d, layers.fully_connected],
        activation_fn=_leaky_relu, normalizer_fn=None,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4 ,4], stride=2)
        net = layers.flatten(net)
        if is_conditional:
            net = tfgan.features.condition_tensor_from_onehot(net, one_hot_labels)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)

        return net


def unconditional_discriminator(img, unused_conditioning, weight_decay=2.5e-5):
    net = _discriminator_helper(img, False, None, weight_decay)
    return layers.linear(net, 1)


def conditional_discreiminator(img, conditioning, weight_decay=2.5e-5):
    _, one_hot_labels = conditioning
    net = _discriminator_helper(img, True, one_hot_labels, weight_decay)
    return layers.linear(net, 1)

