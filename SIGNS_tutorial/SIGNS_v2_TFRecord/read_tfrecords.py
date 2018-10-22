import tensorflow as tf
import os


def read_tfrecords(data_path, batch_size, num_threads, img_size, num_channels):

    # Create a queue to hold filenames. Outputs strings to a queue for an input pipeline.
    # It also has some optional arguments including  num_epochs which indicates the number of epoch
    # you want to to load the data and shuffle which indicates whether to suffle the filenames in the list
    # or not. It is set to True by default.

    assert os.path.isfile(data_path), 'Data path provided does not exist'

    # Create a queue
    print('Creating a queue to hold filenames in a FIFO basis...')
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

    # Define a TFRecords reader
    reader = tf.TFRecordReader()

    # In order to read a record we need to use the following method from the tf.TFRecordReader object
    _, serialized_example = reader.read(filename_queue)

    # Define a decoder: A decoder is needed to decode the record read by the reader. For that we need a
    # serialized example and a dictionary containing the features to extract.
    feature = {'height': tf.FixedLenFeature([], tf.int64),
               'width': tf.FixedLenFeature([], tf.int64),
               'depth': tf.FixedLenFeature([], tf.int64),
               'label': tf.FixedLenFeature([], tf.int64),
               'image': tf.FixedLenFeature([], tf.string)}

    features_parsed = tf.parse_single_example(serialized_example, features=feature)

    # Convert the data from string back to the numbers: tf.decode_raw(bytes, out_type) takes a Tensor
    # of type string and convert it to typeout_type.

    # with tf.Session() as sess:
    #     height = sess.run(tf.cast(features_parsed['height'], tf.int32))
    #     width = sess.run(tf.cast(features_parsed['width'], tf.int32))
    #     depth = sess.run(tf.cast(features_parsed['depth'], tf.int32))
    print('Defining the graph for extracting the data')
    label = tf.cast(features_parsed['label'], tf.int32)
    img_decoded = tf.decode_raw(features_parsed['image'], out_type=tf.float32)

    # Reshape data into its original shape: You should reshape the data (image) into it's original shape
    # before serialization.

    img_reshaped = tf.reshape(img_decoded, [img_size, img_size, num_channels])

    # TODO: Any preprocessing to the images should be done here

    # Batching: Another queue is needed to create batches from the examples. 'Capacity' is the maximum size of the
    # queue. Using more than one thread, it comes up with a faster reading. The first argument in a list of
    # tensors which you want to create batches from.
    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([img_reshaped, label], batch_size=batch_size,
                                            capacity=30, num_threads=num_threads, min_after_dequeue=10)

    # Define the init operations
    init_op_global = tf.global_variables_initializer()
    init_op_local = tf.local_variables_initializer()

    out = {'images': images, 'labels': labels,
           'init_op_global': init_op_global, 'init_op_local': init_op_local}

    return out