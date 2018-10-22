'''Create a set of TFRecords files for training our model'''

import os
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import argparse
import sys


def get_filenames_labels(data_path):
    # Initialize the output lists
    filenames = []
    labels = []

    # Iterate over the 6 different classes
    for i in range(6):
        class_dir = str(i) + '_signs'

        # Get all the filenames of the dir
        filenames_class = os.listdir(os.path.join(data_path, class_dir))
        filenames_class = [os.path.join(data_path, class_dir, f) for f in filenames_class if f.endswith('jpg')]
        # Add the labels of the corresponding class
        labels_class = [i] * len(filenames_class)
        labels = labels + labels_class
        filenames = filenames + filenames_class

    # Shuffle the filenames and labels randomly
    z = list(zip(filenames, labels))
    random.shuffle(z)
    filenames, labels = zip(*z)

    out = {'filenames': filenames, 'labels': labels}

    return out


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_TFRecord(image_paths, labels, out_path, size=(64, 64)):
    # Args:
    # image_paths   List of file-paths for the images.
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.

    print("Creating {} file".format(out_path))

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:

        # Iterate over all the image-paths and class-labels.
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            if i % 100 == 0:
                print('Progress: {}/{} images converted'.format(i, len(labels)))
                sys.stdout.flush()

            # Load the image-file using matplotlib's imread function.
            img = Image.open(path)

            # Resize the image (not activated right now) to the desired size and convert it to a numpy array
            # img = img.resize(size)
            img = np.asarray(img, dtype=np.float32)

            # Get the features to save
            rows = img.shape[0]
            cols = img.shape[1]
            depth = img.shape[2]

            # Convert the image to raw bytes.
            img_bytes = tf.compat.as_bytes(img.tostring())

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = {
                'image': wrap_bytes(img_bytes),
                'label': wrap_int64(label),
                'height': wrap_int64(rows),
                'width': wrap_int64(cols),
                'depth': wrap_int64(depth),
            }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)
            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)
            # Serialize the data.
            serialized = example.SerializeToString()
            # Write the serialized data to the TFRecords file.
            writer.write(serialized)

        print('Progress: {}/{} images converted'.format(i + 1, len(labels)))

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

    train_raw_inputs = get_filenames_labels(os.path.join(args.data_dir, 'train_signs'))
    dev_raw_inputs = get_filenames_labels(os.path.join(args.data_dir, 'dev_signs'))
    test_raw_inputs = get_filenames_labels(os.path.join(args.data_dir, 'test_signs'))

    convert_to_TFRecord(
        image_paths=train_raw_inputs['filenames'],
        labels=train_raw_inputs['labels'],
        out_path=os.path.join(args.out_dir, 'train.tfrecords')
    )

    convert_to_TFRecord(
        image_paths=dev_raw_inputs['filenames'],
        labels=dev_raw_inputs['labels'],
        out_path=os.path.join(args.out_dir, 'dev.tfrecords')
    )

    convert_to_TFRecord(
        image_paths=test_raw_inputs['filenames'],
        labels=test_raw_inputs['labels'],
        out_path=os.path.join(args.out_dir, 'test.tfrecords')
    )

    print('TFRecord converter process finished.')