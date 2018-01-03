import numpy as np
import tensorflow as tf
import gan_study.NetWorks as network
import tensorflow.contrib.gan as tfgan
import gan_study.data_provider as data_provider
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_study import load
import scipy.misc
import os

train_samples, train_labels = load.load_train_data()

flags = tf.flags

flags.DEFINE_integer('batch_size', 32, 'The number of images i each train batch')

flags.DEFINE_integer('max_number_of_steps', 2000, 'The maximum number of gradient steps.')
flags.DEFINE_integer('noise_dims', 64, 'Dimensions of the generator noise vector')
flags.DEFINE_string('dataset_dir', 'D:\code\Python\Library_code\gan_study\data', 'Location of data')
flags.DEFINE_string('eval_dir', 'D:\code\Python\Library_code\gan_study\mnist-estimator', 'Directory where the results are saved to.')

FLAGS = flags.FLAGS

def _get_train_input_fn(batch_size, noise_dims, dataset_dir = None, num_threads = 4):
    def train_input_fn():
        with tf.device('/cpu:0'):
            images, _ = data_provider.provide_data(
                'train', batch_size, dataset_dir, num_threads=num_threads
            )
            noise = tf.random_normal([batch_size, noise_dims])
        return noise, images
    return train_input_fn


def get_train_input_fn(batch_size, noise_dims):
    def train_input_fn():
        images = tf.cast(train_samples, tf.float32)
        input_queue = tf.train.slice_input_producer([images], shuffle=False)
        images_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1)
        noise = tf.random_normal([batch_size, noise_dims])
        return noise, images_batch
    return train_input_fn



def _get_predict_input_fn(batch_size, noise_dims):
    def predict_input_fn():
        noise = tf.random_normal([batch_size, noise_dims])
        return noise
    return predict_input_fn


def main(_):
    gan_estimator = tfgan.estimator.GANEstimator(
        generator_fn=network.unconditional_generator,
        discriminator_fn=network.unconditional_discriminator,
        generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
        discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
        generator_optimizer=tf.train.AdamOptimizer(0.001, 0.5),
        discriminator_optimizer=tf.train.AdamOptimizer(0.0001, 0.5),
        add_summaries=tfgan.estimator.SummaryType.IMAGES

    )
    train_input_fn = get_train_input_fn(FLAGS.batch_size, FLAGS.noise_dims)
    gan_estimator.train(train_input_fn, max_steps=FLAGS.max_number_of_steps)

    predict_input_fn = _get_predict_input_fn(36, FLAGS.noise_dims)
    predict_iterable = gan_estimator.predict(predict_input_fn)
    print(type(predict_iterable))
    predictions = [predict_iterable.__next__() for _ in range(36)]

    images_rows = [np.concatenate(predictions[i:i+6], axis=0) for i in range(0, 36, 6)]
    tiled_image = np.concatenate(images_rows, axis=1)
    print(tiled_image.shape)

    if not tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.MakeDirs(FLAGS.eval_dir)
    scipy.misc.imsave(os.path.join(FLAGS.eval_dir, 'unconditional_gan.png'), np.squeeze(tiled_image, axis=2))


if __name__ == "__main__":
    tf.app.run()


