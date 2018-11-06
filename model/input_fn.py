import tensorflow as tf
import glob
import numpy as np
import os


def preprocess(x, mean_channels, is_training):
    # Subtract the overall mean of the images
    # x = x - tf.reshape(mean_channels, [1, 1, 3])
    x = x / tf.constant(255, dtype=tf.float32)
    return x


def _parse_function(example_proto, mean_channels, is_training):
    ''' Parse the tfrecords files

    :param example_proto:
    :param mean_channels:
    :param is_training:
    :return:
    '''

    keys_to_features = {
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
    }

    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    flatten_image = tf.decode_raw(parsed_features['image'], tf.float32)

    # TODO: Ask how not to hardcode these values
    shape = tf.stack([512, 512, 3])
    # shape = tf.stack([parsed_features['height'],
    #                   parsed_features['width'],
    #                   parsed_features['depth']])

    img = tf.reshape(flatten_image, shape)
    label = parsed_features['label']

    # TODO: Ask about preprocessing.
    img = preprocess(img, mean_channels, is_training)

    return img, label


def input_fn(tfrecord_dir, mean_npz, n_images=None, is_training=False, seed=None,
             name='Unknown', params=None):
    '''

    :param tfrecord_dir:
    :param mean_npz:
    :param n_images:
    :param is_training:
    :param seed:
    :param name:
    :param params:
    :return:
    '''

    # Create the pattern for the filenames
    filenames_pattern = tfrecord_dir + '/*balanced.tfrecords'
    filenames = glob.glob(filenames_pattern)
    n_shards = len(filenames)

    assert n_shards != 0, 'Error: No filenames found'

    # Count the number of images inside the directory if necessary
    if not n_images:
        n_images = 0
        for filename in filenames:
            print('Counting images from tfrecord {}'.format(filename))
            n_images_file = sum(1 for _ in tf.python_io.tf_record_iterator(filename))
            print('Number of images found: {}'.format(n_images_file))
            n_images += n_images_file

    # Compute the number of iterations needed as a function of the number of files and the batch size
    n_iter_per_epoch = n_images//params.batch_size if n_images % params.batch_size == 0 else n_images//params.batch_size+1
    print("\n{} contains {} images. Batch size is {}. We would need {} iterations/epoch.".format(
        name.upper(),
        n_images,
        params.batch_size,
        n_iter_per_epoch
    ))

    # Create a dataset formed by all tfrecord files
    files = tf.data.Dataset().list_files(tf.constant(filenames_pattern))

    # TODO: Ask about mean channels and mean_npz files
    # mean_channels from mean_npz file
    # if not os.path.exists(mean_npz):
    #     raise Exception("mean_channels file does not exists: {}".format(
    #         mean_npz
    #     ))
    # mean_channels = tf.constant(np.load(mean_npz).tolist())
    mean_channels = None

    # Create TFRecordDataset objects from each of the tfrecord filenames
    dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=10)
    # TODO: Ask what does interleave exactly and why map does not work here?

    # Parse the record into tensors + preprocessing
    dataset = dataset.map(map_func=lambda x: _parse_function(x, mean_channels, is_training), num_parallel_calls=10)

    # Shuffle the data if training
    if is_training:
        dataset = dataset.shuffle(buffer_size=n_images)

    # Define the batch size, make it repeatable and establish a prefetch size
    dataset = dataset.batch(params.batch_size).repeat().prefetch(buffer_size=20)

    # Create the corresponding iterator
    iterator = dataset.make_initializable_iterator()

    # Get the needed operations ready
    it_init_op = iterator.initializer
    images, labels = iterator.get_next()

    output = {'images': images, 'labels': labels, 'it_init_op': it_init_op, 'n_iter_per_epoch': n_iter_per_epoch}

    return output


