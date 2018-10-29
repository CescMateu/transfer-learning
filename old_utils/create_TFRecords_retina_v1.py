'''Create a set of TFRecords files for training our model'''

import glob
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import argparse
import sys
import os


def get_labels_from_illness(illness, n_filenames):
    '''
    Returns a list with the id of the illness and length of n_filenames
    '''

    assert illness in ['normal', 'altpig', 'dmae', 'excavation', 'membrana',
                       'nevus'], 'The introduced illness does not exist'
    assert n_filenames != 0, 'The number of filenames should be greater than 0'

    if illness == 'normal':
        illness_id = 0
    elif illness == 'altpig':
        illness_id = 1
    elif illness == 'dmae':
        illness_id = 2
    elif illness == 'excavation':
        illness_id = 3
    elif illness == 'membrana':
        illness_id = 4
    elif illness == 'nevus':
        illness_id = 5
    else:
        raise ValueError('The introduced illness "{}" does not exist.'.format(illness))

    return [illness_id] * n_filenames


def get_filenames_labels_mode(parent_dir, mode_to_retrieve, shuffle=False):
    '''
    Given the parent_dir and the mode ('train', 'test', 'validation'), this function returns a dictionary with the names of
    all the filenames found inside the parent directory and the corresponding labels
    '''

    assert os.path.isdir(parent_dir), 'The parent directory specified does not exist'
    assert mode_to_retrieve in ['train', 'test', 'validation'], 'The specified mode does not exist. Options: "train", "test", "validation"'

    illness_dirs = glob.glob(parent_dir + '/*')  # illness level
    filenames = []  # create an empty list for saving the filenames
    labels = []  # create an empty list for saving the labels

    for illness_dir in illness_dirs:

        mode_dirs = glob.glob(illness_dir + '/*')  # mode level

        for mode_dir in mode_dirs:
            mode = mode_dir.split('/')[-1]  # retrieve the last part of the pathname (the mode)
            if mode == mode_to_retrieve:
                class_dirs = glob.glob(mode_dir + '/*')  # class level

                for class_dir in class_dirs:
                    illness = class_dir.split('/')[-1]  # get which illness are we processing now

                    # Retrieve the filenames and corresponding labels
                    filenames_to_retrieve = [f for f in glob.glob(class_dir + '/*') if f.endswith('.jpg')]
                    labels_to_retrieve = get_labels_from_illness(illness=illness,
                                                                 n_filenames=len(filenames_to_retrieve))

                    assert len(filenames_to_retrieve) == len(
                        labels_to_retrieve), 'The lengths of the retrieve filenames and labels do not coincide.'

                    # Append the previous results to the overall lists
                    filenames = filenames + filenames_to_retrieve
                    labels = labels + labels_to_retrieve

    # Check everything is going smooth
    assert len(filenames) == len(labels), 'The lengths of the total filenames and labels do not coincide.'

    # Shuffle the lists randomly if specified
    if shuffle:
        z = list(zip(filenames, labels))
        random.shuffle(z)
        filenames, labels = zip(*z)

    out = {'filenames': filenames, 'labels': labels}

    return out


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_list_to_TFRecord(image_paths, labels, output_name, output_size_files=200, resize=False, new_size=None):
    '''
    Args:
    image_paths           List of file-paths for the images.
    labels                Class-labels for the images.
    output_name           File-name for the TFRecords output file, without the extension (.tfrecords)
    output_size_files     Number of images saved in each .tfrecords file
    resize                Boolean indicating whether we want the images resized before saving them into TFRecords
    new_size              Integer indicating the new size of the images. Deprecated if resize = False
    '''

    assert len(image_paths) == len(labels), 'Number of image paths and labels do not coincide.'

    # Compute the number of needed TFRecords files
    n_files_created = 0
    n_files_needed = len(image_paths) // output_size_files + 1
    n_written_images = 0

    # Iterate over the diferent TFRecords files that will be created
    for file_id in range(n_files_needed):
        output_file_name = output_name + '_' + str(file_id) + '.tfrecords'
        print("Creating new file: {}".format(output_file_name))
        n_files_created += 1

        # Retrieve the next batch of size 'output_size_files' of filenames and labels
        initial_image_id = file_id * output_size_files
        final_image_id = (file_id + 1) * output_size_files
        image_paths_to_be_coded = image_paths[initial_image_id:final_image_id]
        labels_to_be_coded = labels[initial_image_id:final_image_id]

        # Bug-free code
        assert len(image_paths_to_be_coded) == len(
            labels_to_be_coded), 'The next batch of filenames, labels has a mismatch of lenghts'

        # Open a new TFRecordWriter to write the filenames and labels into a TFRecords file
        with tf.python_io.TFRecordWriter(output_file_name) as writer:

            # Iterate over all the image-paths and class-labels.
            for i, (path, label) in enumerate(zip(image_paths_to_be_coded, labels_to_be_coded)):

                # Load the image-file using matplotlib's imread function.
                img = Image.open(path)
                # Resize the image to the desired size
                if resize:
                    img = img.resize(size=new_size, resample=Image.BILINEAR)
                # Transform the image into a numpy array and get the features to save
                img = np.asarray(img, dtype=np.float32)
                rows, cols, depth = img.shape[0], img.shape[1], img.shape[2]
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
                # Update the number of written images for printing the process
                n_written_images += 1

        # Print the progress at the end of every file and flush the memory
        print('Progress: {}/{} images converted'.format(n_written_images, len(labels)))
        # Check that the created file has the expected amount of files
        len_tfrecord = len([x for x in tf.python_io.tf_record_iterator(output_file_name)])
        assert len_tfrecord == len(image_paths_to_be_coded), 'Expected size of the TFRecord file does not coincide with the input size of the list of filenames'
        # Flush the memory
        sys.stdout.flush()


if __name__ == '__main__':

    # Create an ArgumentParser object to be able to specify some parameters when executing the file
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
                        help="Directory containing the original datasets")
    parser.add_argument('--out_dir', default=None,
                        help="Target directory")
    parser.add_argument('--n_img_per_file', default=200,
                        help="Maximum number of images that will be saved in each TFRecords file")

    # Get the data_dir from the ArgParser object
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), 'The provided path does not exist'

    # Get the filenames and labels from the files
    train_inputs = get_filenames_labels_mode(parent_dir=args.data_dir,
                                             mode_to_retrieve='train', shuffle=True)
    validation_inputs = get_filenames_labels_mode(parent_dir=args.data_dir,
                                                  mode_to_retrieve='validation', shuffle=True)
    test_inputs = get_filenames_labels_mode(parent_dir=args.data_dir,
                                            mode_to_retrieve='test', shuffle=True)

    print('----- Creating train files')
    convert_list_to_TFRecord(
        image_paths=train_inputs['filenames'],
        labels=train_inputs['labels'],
        output_name=(os.path.join(args.out_dir, 'train/' 'train')),
        output_size_files=args.n_img_per_file,
        resize=False
    )

    print('----- Creating validation files')
    convert_list_to_TFRecord(
        image_paths=validation_inputs['filenames'],
        labels=validation_inputs['labels'],
        output_name=(os.path.join(args.out_dir, 'validation/' 'validation')),
        output_size_files=args.n_img_per_file,
        resize=False
    )

    print('----- Creating test files')
    convert_list_to_TFRecord(
        image_paths=test_inputs['filenames'],
        labels=test_inputs['labels'],
        output_name=(os.path.join(args.out_dir, 'test/' 'test')),
        output_size_files=args.n_img_per_file,
        resize=False
    )

    print('TFRecord converter process finished.')


