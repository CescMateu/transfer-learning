"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf


def _parse_function(serialized):
    features = \
        {
            'image': tf.FixedLenFeature((), tf.string, default_value=''),
            'label': tf.FixedLenFeature((), tf.int64, default_value=0),
            'height': tf.FixedLenFeature((), tf.int64, default_value=0),
            'width': tf.FixedLenFeature((), tf.int64, default_value=0),
            'depth': tf.FixedLenFeature((), tf.int64, default_value=0),
        }
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)
    # Get the image as raw bytes and decode it
    image_raw = tf.decode_raw(parsed_example['image'], tf.float32)
    shape = tf.stack([64, 64, 3])

    image = tf.reshape(image_raw, shape)
    label = tf.cast(parsed_example['label'], tf.int32)

    # image = tf.subtract(image, 116.779) # Zero-center by mean pixel
    # image = tf.reverse(image, axis=[2])  # 'RGB'->'BGR'

    return image, label


def _train_preprocess(image, label, use_random_flip):
    """Image preprocessing for training.

    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """
    if use_random_flip:
        image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def input_fn(tfrecord_filenames, num_records, params):
    """Input function for the SIGNS dataset.

    The filenames have format "{label}_IMG_{id}.jpg".
    For instance: "data_dir/2_IMG_4584.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """

    dataset = tf.data.TFRecordDataset(tfrecord_filenames)
    dataset = dataset.map(_parse_function)  # Parse the record into tensors.
    dataset = dataset.shuffle(num_records) # Shuffle the dataset
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.batch(params.batch_size).prefetch(params.batch_size) # Define the batch and prefetch sizes
    iterator = dataset.make_initializable_iterator()

    # Get the needed operations ready
    it_init_op = iterator.initializer
    images, labels = iterator.get_next()

    output = {'images': images, 'labels': labels, 'it_init_op': it_init_op}

    return output

