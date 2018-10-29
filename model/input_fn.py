import tensorflow as tf
import numpy as np
import os
import glob

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
    label = tf.expand_dims(parsed_features['label'], axis=-1)

    # TODO: Ask about preprocessing.
    #img = preprocess(img, mean_channels, is_training)

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

    # Count the number of images inside the directory if the number is not passed
    if not n_images:
        n_images = 0
        for filename in filenames:
            print('Counting images from tfrecord {}'.format(filename))
            n_images += sum(1 for _ in tf.python_io.tf_record_iterator(filename))

    # Compute the number of iterations needed as a function of the number of files and the batch size
    n_iter_per_epoch = n_images//params.batch_size if n_images % params.batch_size == 0 else n_images//params.batch_size+1
    print("{} contains {} images --> using batch size of {}, so we got {} iterations/epoch.".format(
        name,
        n_images,
        params.batch_size,
        n_iter_per_epoch
    ))

    # Create a dataset formed by all tfrecord files and shuffle them
    files = tf.data.Dataset().list_files(tf.constant(filenames_pattern)).shuffle(n_shards)

    # TODO: Ask about mean channels and mean_npz files
    # mean_channels from mean_npz file
    # if not os.path.exists(mean_npz):
    #     raise Exception("mean_channels file does not exists: {}".format(
    #         mean_npz
    #     ))
    # mean_channels = tf.constant(np.load(mean_npz).tolist())
    mean_channels = None

    # Interleave shuffled tfrecord files, create tfrecord datasets from each file and parse/preprocess them
    # dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=4)
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=10
        )
    )

    # Shuffle and repeat the dataset if training
    if is_training:
        print('Shuffling buffer size: {}'.format(n_images//n_shards*2))
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(
                buffer_size=n_images//n_shards*2,
                seed=seed
            )
        )
    else:
        dataset = dataset.repeat()

    # Parse the record into tensors + preprocessing
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            map_func=lambda x: _parse_function(x, mean_channels, is_training),
            batch_size=params.batch_size,
            num_parallel_batches=10,
        )
    )

    # Prefetching
    dataset = dataset.prefetch(buffer_size=20)

    # Create the corresponding iterator
    iterator = dataset.make_initializable_iterator()

    # Get the needed operations ready
    it_init_op = iterator.initializer
    images, labels = iterator.get_next()

    output = {'images': images, 'labels': labels, 'it_init_op': it_init_op, 'n_iter_per_epoch': n_iter_per_epoch}

    return output


# def horizontal_flip(x):
#     rand = tf.random_uniform(shape=(1,))[0]
#     cond = tf.greater(rand, tf.constant(0.5))
#     x = tf.cond(cond, lambda: tf.image.flip_left_right(x), lambda: x)
#     return x
#
#
# def random_crop(x):
#     x = tf.random_crop(x, tf.constant([224, 224, 3]))
#     return x
#
#
# def central_crop(x):
#     # Fraction of central image to be kept along each dimension is 224/256 = 0.875
#     x = tf.image.central_crop(x, 0.875)
#     return x
#
#
# def preprocess(x, mean_channels, is_training):
#     if is_training:
#         x = horizontal_flip(x)
#         x = random_crop(x)
#     else:
#         x = central_crop(x)
#     x = x - tf.reshape(mean_channels, [1, 1, 3])
#     x = x / tf.constant(255, dtype=tf.float32)
#     return x
