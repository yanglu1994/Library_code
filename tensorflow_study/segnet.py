import tensorflow_study.load as load
from tensorflow_study.unet import Network

if __name__ == "__main__":
    train_samples, train_labels = load.load_train_data()
    test_samples, test_labels = load.load_test_data()

    TRAIN_BATCH_SIZE = 1
    TEST_BATCH_SIZE = 1
    DROPOUT = False

    print('Training set', train_samples.shape, train_labels.shape)
    print('    Test set', test_samples.shape, test_labels.shape)

    image_size = load.image_size
    num_channels = load.num_channels

    def train_data_iterator(samples, labels, iteration_steps, chunkSize):
        if len(samples) != len(labels):
            raise Exception('Length of samples and labels must equal')
        stepStart = 0  # initial step
        i = 0
        while i < iteration_steps:
            stepStart = (i * chunkSize) % (labels.shape[0] - chunkSize)
            yield i, samples[stepStart:stepStart + chunkSize], labels[stepStart:stepStart + chunkSize]
            i += 1

    def test_data_iterator(samples, labels, chunkSize):
        if len(samples) != len(labels):
            raise Exception('Length of samples and labels must equal')
        stepStart = 0  # initial step
        i = 0
        while stepStart < len(samples):
            stepEnd = stepStart + chunkSize
            if stepEnd < len(samples):
                yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
                i += 1
            stepStart = stepEnd

    net = Network(
        train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE,
        dropout_rate=0.4, base_learning_rate=0.03, decay_rate=0.90
    )

    train_input, test_input = net.define_inputs(
        train_samples_shape=(TRAIN_BATCH_SIZE, image_size, image_size, num_channels),
        train_labels_shape=(TRAIN_BATCH_SIZE, image_size, image_size, 2*num_channels),
        test_samples_shape=(TEST_BATCH_SIZE, image_size, image_size, num_channels),
        test_labels_shape=(TEST_BATCH_SIZE, image_size, image_size, 2*num_channels)
    )

    conv1_1 = net.add_conv(data_flow=test_input, patch_size=3, in_depth=num_channels, out_depth=64, activation='relu', name='conv1_1')
    conv1_2 = net.add_conv(data_flow=conv1_1, patch_size=3, in_depth=64, out_depth=64, activation='relu', name='conv1_2')
    pool1 = net.add_pool(data_flow=conv1_2, pooling_scale=2, name='pool1')

    conv2_1 = net.add_conv(data_flow=pool1, patch_size=3, in_depth=64, out_depth=128, activation='relu', name='conv2_1')
    conv2_2 = net.add_conv(data_flow=conv2_1, patch_size=3, in_depth=128, out_depth=128, activation='relu', name='conv2_1')
    pool2 = net.add_pool(data_flow=conv2_2, pooling_scale=2, name='pool2')

    conv3_1 = net.add_conv(data_flow=pool2, patch_size=3, in_depth=128, out_depth=256, activation='relu', name='conv3_1')
    conv3_2 = net.add_conv(data_flow=conv3_1, patch_size=3, in_depth=256, out_depth=256, activation='relu', name='conv3_2')
    conv3_3 = net.add_conv(data_flow=conv3_2, patch_size=3, in_depth=256, out_depth=256, activation='relu',name='conv3_3')
    pool3 = net.add_pool(data_flow=conv3_3, pooling_scale=2, name='pool3')

    conv4_1 = net.add_conv(data_flow=pool3, patch_size=3, in_depth=256, out_depth=512, activation='relu', name='conv4_1')
    conv4_2 = net.add_conv(data_flow=conv4_1, patch_size=3, in_depth=512, out_depth=512, activation='relu', name='conv4_2')
    conv4_3 = net.add_conv(data_flow=conv4_2, patch_size=3, in_depth=512, out_depth=512, activation='relu', name='conv4_3')
    pool4 = net.add_pool(data_flow=conv4_3, pooling_scale=2, name='pool4')

    conv5_1 = net.add_conv(data_flow=pool4, patch_size=3, in_depth=512, out_depth=512, activation='relu', name='conv5_1')
    conv5_2 = net.add_conv(data_flow=conv5_1, patch_size=3, in_depth=512, out_depth=512, activation='relu', name='conv5_2')
    conv5_3 = net.add_conv(data_flow=conv5_2, patch_size=3, in_depth=512, out_depth=512, activation='relu', name='conv5_3')
    pool5 = net.add_pool(data_flow=conv5_3, pooling_scale=2, name='pool5')

    us1 = net.add_us(data_flow=pool5, patch_size=3, in_channels=512, out_channels=512, output_shape=[TEST_BATCH_SIZE, 25, 25, 512], name='us1')
    conv6_1= net.add_conv(data_flow=us1, patch_size=3, in_depth=512, out_depth=512, activation='relu', name='conv6_1')
    conv6_2 = net.add_conv(data_flow=conv6_1, patch_size=3, in_depth=512, out_depth=512, activation='relu', name='conv6_2')
    conv6_3 = net.add_conv(data_flow=conv6_2, patch_size=3, in_depth=512, out_depth=512, activation='relu', name='conv6_3')

    us2 = net.add_us(data_flow=conv6_3, patch_size=3, in_channels=512, out_channels=512, output_shape=[TEST_BATCH_SIZE, 50, 50, 512], name='us2')
    conv7_1 = net.add_conv(data_flow=us2, patch_size=3, in_depth=512, out_depth=512, activation='relu', name='conv7_1')
    conv7_2 = net.add_conv(data_flow=conv7_1, patch_size=3, in_depth=512, out_depth=512, activation='relu', name='conv7_2')
    conv7_3 = net.add_conv(data_flow=conv7_2, patch_size=3, in_depth=512, out_depth=256, activation='relu', name='conv7_3')

    us3 = net.add_us(data_flow=conv7_3, patch_size=3, in_channels=256, out_channels=256, output_shape=[TEST_BATCH_SIZE, 100, 100, 256], name='us3')
    conv8_1 = net.add_conv(data_flow=us3, patch_size=3, in_depth=256, out_depth=256, activation='relu', name='conv8_1')
    conv8_2 = net.add_conv(data_flow=conv8_1, patch_size=3, in_depth=256, out_depth=256, activation='relu', name='conv8_2')
    conv8_3 = net.add_conv(data_flow=conv8_2, patch_size=3, in_depth=256, out_depth=128, activation='relu', name='conv8_3')

    us4 = net.add_us(data_flow=conv8_3, patch_size=3, in_channels=128, out_channels=128, output_shape=[TEST_BATCH_SIZE, 200, 200, 128], name='us4')
    conv9_1 = net.add_conv(data_flow=us4, patch_size=3, in_depth=128, out_depth=128, activation='relu', name='conv9_1')
    conv9_2 = net.add_conv(data_flow=conv9_1, patch_size=3, in_depth=128, out_depth=64, activation='relu', name='conv9_2')

    us5 = net.add_us(data_flow=conv9_2, patch_size=3, in_channels=64, out_channels=64, output_shape=[TEST_BATCH_SIZE, 400, 400, 64], name='us5')
    conv10_1 = net.add_conv(data_flow=us5, patch_size=3, in_depth=64, out_depth=64, activation='relu', name='conv10_1')
    conv10_2 = net.add_conv(data_flow=conv10_1, patch_size=3, in_depth=64, out_depth=2, name='conv10_2')


    net.define_model(conv10_2)


    # net.train(train_samples, train_labels, data_iterator=train_data_iterator, iteration_steps=1000)
    net.test(test_samples, test_labels, data_iterator=test_data_iterator)



