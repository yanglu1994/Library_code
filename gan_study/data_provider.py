import tensorflow as tf

import tensorflow.contrib.slim as slim
from gan_study import dataset_factory as datasets

def provide_data(split_name, batch_size, dataset_dir, num_readers = 1, num_threads = 1):
    dataset = datasets.get_dataset('mnist', split_name, dataset_dir=dataset_dir)
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        common_queue_capacity=2*batch_size,
        common_queue_min=batch_size,
        shuffle=(split_name == 'train')
    )
    [image, label] = provider.get(['image', 'label'])

    image = (tf.to_float(image) - 128.0) / 128.0

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=5*batch_size
    )

    # one_hot_labels = tf.one_hot(labels, dataset.num_class)

    return images,  dataset.num_samples