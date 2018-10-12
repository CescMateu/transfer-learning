import tensorflow as tf

def _parse_fn(filename, label, size = None):
    '''
    Obtain the image from the filename (for both training and validation).

        The following operations are applied:
            - Decode the image from jpeg format
            - Convert to float and to range [0, 1]

    :param filename: (string) path with the location of the image
    :param label: (int) class of the image (0, 1, 2, 3, 4, 5)
    :param size: [Optional] (int or None) if not 'None', output size of the image will be (size x size)

    :return: (dict) Dictionary containing the tf.data.Datasets for the inputs, labels and the corresponding iterator
    '''


    if size is not None:
        assert isinstance(size, int), 'If not None, size must be an integer'

    # Read the image and decode it in the RGB mode
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)

    # Resize the image in the desired size
    if size is not None:
        image = tf.image.resize_images(image, size=[size, size])

    return image, label

def train_preprocess_fn(f, l):
    # TODO: Complete this function.
    # Ideas:
    # - Compute the mean image and subtract it from all the images
    # - Image augmentation
    # - ...

    return f, l

def input_fn(filenames, labels, params, is_training):
    '''

    TODO: Finish documentation

    :param is_training:
    :param filenames:
    :param labels:
    :param params:
    :return:
    '''

    assert len(filenames) == len(labels), 'Filenames and labels must have the same length'

    num_samples = len(filenames)

    # Create the parse and preprocessing functions
    parse_fn = lambda f, l: _parse_fn(f, l, None)
    train_preproc_fn = lambda f, l: train_preprocess_fn(f, l)

    # Create the tf.data.Dataset. Shuffling before the parsing operation on the images is much better,
    # because it's much more light computationally.
    # To summarize, one good order for the different transformations is:
    # - Create the dataset
    # - Shuffle(with a big enough buffer size)
    # - Repeat
    # - Map with the actual work (preprocessing, augmentation…) using multiple parallel calls
    # - Batch
    # - Prefetch

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                   .shuffle(num_samples)  # putting the whole dataset into the buffer ensures good shuffling
                   .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
                   .map(train_preproc_fn, num_parallel_calls=params.num_parallel_calls) # preprocessing function only for training
                   .batch(params.batch_size)
                   .prefetch(1) # make sure you always have one batch ready to serve
                   .repeat() # TODO: This .repeat() should not be necessary. Check the num_steps in training/train()
                   )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                   .shuffle(num_samples)
                   .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
                   .batch(params.batch_size)
                   .prefetch(1)
                   )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    # Put everything together in a dictionary
    inputs = {'images': images, 'labels': labels, 'iterator': iterator_init_op}

    return inputs




