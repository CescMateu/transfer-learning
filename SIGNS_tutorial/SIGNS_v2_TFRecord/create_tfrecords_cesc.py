'''Script that transforms a directory of images into a TFRecords format.'''

import random
import os
import math
import cv2
import numpy as np
import tensorflow as tf
import sys
import argparse


# Define a function to get all the filenames and labels in a directory and perform a train/dev/test split
def get_filenames_labels_from(dir, train_fraction, test_fraction, image_ext, shuffle=True):
    '''

    :param dir:
    :param train_fraction:
    :param test_fraction:
    :param image_ext:
    :param shuffle:
    :return:
    '''

    assert (train_fraction + test_fraction) < 1, 'Train/Test fractions must be less than 1'
    assert image_ext[0] == '.', 'Image extension should start with a dot (ex: ".jpg")'

    # Get all the filenames that end with the specified image extension
    filenames = [f for f in os.listdir(dir) if f.endswith(image_ext)]

    # Get the labels (This may change for every project. Depends on how the labels are specified)
    labels = [int(f[0]) for f in filenames]

    # Add the rest of the path to the filenames
    filenames = [dir + f for f in filenames]

    # Zip the two components
    dt = list(zip(filenames, labels))

    # Shuffle if necessary
    if shuffle:
        random.shuffle(dt)

    # Split data in train, dev and test
    n = len(dt)
    print('Total number of examples is {}'.format(n))

    # Create a 60/20/20 splitting
    train = dt[:math.floor(train_fraction * n)]
    dev = dt[math.floor(train_fraction * n):math.floor((test_fraction + train_fraction) * n)]
    test = dt[math.floor((test_fraction + train_fraction) * n):]

    print('Train size is {}, dev size is {}, test size is {}.'.format(len(train), len(dev), len(test)))

    # Wrap everything in a dict
    out = {'train':train, 'dev': dev, 'test':test}

    return out


# Define a function that takes an image path and returns the corresponding resized np.array
def load_img(image_path, image_output_size):
    # read an image and resize to (output_size, output_size)
    # cv2 load images as BGR, convert it to RGB

    img = cv2.imread(image_path)
    #img = cv2.resize(img, (image_output_size, image_output_size), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)

    return img


# Define the basic TFRecords functions
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_TFRecord(dataset, output_file_name, out_dir):

    filename_out = output_file_name + '.tfrecords'  # address to save the TFRecords file
    print('Output TFRecords file: ' + filename_out)
    path_out = os.path.join(out_dir, filename_out)

    # Open the TFRecords file
    writer = tf.python_io.TFRecordWriter(path_out)

    for i in range(len(dataset)):
        # Print how many images are saved every 100 images
        if i % 100 == 0:
            print('Images saved: {}/{}'.format(i, len(dataset)))
            sys.stdout.flush()

        # Load an image and label
        img = load_img(dataset[i][0], image_output_size=64)
        label = dataset[i][1]

        # Get the features to save
        rows = img.shape[0]
        cols = img.shape[1]
        depth = img.shape[2]

        # Create a feature
        feature = {'height': _int64_feature(rows),
                   'width': _int64_feature(cols),
                   'depth': _int64_feature(depth),
                   'label': _int64_feature(label),
                   'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


if __name__ == '__main__':

    # Create an ArgumentParser object to be able to specify some parameters when executing the file
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
                        help="Directory containing the original datasets")
    parser.add_argument('--out_dir', default=None,
                        help="Target directory")

    # Get the data_dir from the ArgParser object
    args = parser.parse_args()

    assert os.path.exists(args.data_dir), 'The provided path does not exist'

    # Set a seed for reproducibility
    random.seed(1234)

    print('Retreiving filenames and labels from the specified directory...')
    filenames_labels_dict = get_filenames_labels_from(args.data_dir, 0.6, 0.2, '.jpg')

    print('Initialising TFRecord converter...')
    for mode in ['train', 'dev', 'test']:
        convert_to_TFRecord(filenames_labels_dict[mode], mode, args.out_dir)

    print('TFRecord converter process finished.')
