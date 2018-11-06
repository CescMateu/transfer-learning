'''Create a set of TFRecords files for training our model'''

import glob
import tensorflow as tf
from PIL import Image
import numpy as np
import random
import argparse
import sys
import os
import matplotlib.pyplot as plt


def get_labels_from_illness(illness, n_filenames):
    '''
    Returns a list with the id of the illness and length of n_filenames
    '''
    
    assert illness in ['normal', 'altpig', 'dmae', 'excavation', 'membrana', 'nevus'], 'The introduced illness ' \
                                                                                       'does not exist'
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
    
    return [illness_id]*n_filenames


def get_filenames_labels_mode_pathologies(parent_dir, mode_to_retrieve, class_type_to_retrieve):
    '''
    Given the parent_dir, the mode ('train', 'test', 'validation') and the class label in string format,
    this function returns a dictionary with the names of all the filenames found inside the parent directory
    with that mode and class label
    '''
    
    assert os.path.isdir(parent_dir), 'The parent directory specified does not exist'
    assert mode_to_retrieve in ['train', 'test', 'validation'], 'The specified mode does not exist. ' \
                                                                'Options: "train", "test", "validation"'
    assert class_type_to_retrieve in ['altpig', 'dmae', 'excavation', 'membrana', 'nevus']
    
    # Define the pathology dirname
    illness_dir = 'u_{}_symbolic_512'.format(class_type_to_retrieve)
    dir_to_search = os.path.join(parent_dir, illness_dir, mode_to_retrieve, class_type_to_retrieve)
    # Example: dir_to_search = 'data/retina_data_susbset/u_altpig_symbolic_512/train/altpig'

    # Retrieve the filenames and corresponding labels
    filenames_to_retrieve = [f for f in glob.glob(dir_to_search + '/*') if f.endswith('.jpg')]
    labels_to_retrieve = get_labels_from_illness(illness=class_type_to_retrieve, n_filenames=len(filenames_to_retrieve))

    assert len(filenames_to_retrieve) == len(labels_to_retrieve), 'The lengths of the retrieved filenames and labels ' \
                                                                  'do not coincide.'
    
    # Give some feedback
    print('Retrieved {} images with label "{}"'.format(len(filenames_to_retrieve), class_type_to_retrieve))

    out = {'filenames': filenames_to_retrieve, 'labels': labels_to_retrieve}
    
    return out


def get_filenames_labels_mode_healthy(parent_dir, mode_to_retrieve):
    '''
    Given the parent_dir, the mode ('train', 'test', 'validation'), this function returns a dictionary with the 
    filenames of 'normal' class found inside the parent directory with that mode.
    '''
    
    assert os.path.isdir(parent_dir), 'The parent directory specified does not exist'
    assert mode_to_retrieve in ['train', 'test', 'validation'], 'The specified mode does not exist. ' \
                                                                'Options: "train", "test", "validation"'

    filenames = []  # create an empty list for saving the filenames
    labels = []  # create an empty list for saving the labels
    pathologies = ['altpig', 'dmae', 'excavation', 'membrana', 'nevus']
    pathologies_dirs = [os.path.join(parent_dir, 'u_{}_symbolic_512'.format(p)) for p in pathologies]
    
    for pathology_dir in pathologies_dirs:
        mode_dirs = glob.glob(pathology_dir + '/*')  # mode level
        
        for mode_dir in mode_dirs:
            mode = mode_dir.split('/')[-1]  # retrieve the last part of the pathname (the mode)
            if mode == mode_to_retrieve:
                dir_to_search = mode_dir + '/normal/*'  # We only want the 'normal' images in this function
                
                # Retrieve the filenames and corresponding labels
                filenames_to_retrieve = [f for f in glob.glob(dir_to_search) if f.endswith('.jpg')]
                labels_to_retrieve = get_labels_from_illness(illness='normal', n_filenames=len(filenames_to_retrieve))

                assert len(filenames_to_retrieve) == len(labels_to_retrieve), 'The lengths of the retrieve filenames' \
                                                                              ' and labels do not coincide.'

                # Append the previous results to the overall lists
                filenames = filenames + filenames_to_retrieve
                labels = labels + labels_to_retrieve

    # Check everything is going smooth
    assert len(filenames) == len(labels), 'The lengths of the total filenames and labels do not coincide.'
    
    # Give some feedback
    print('Retrieved {} images with label "normal"'.format(len(filenames)))
    out = {'filenames': filenames, 'labels': labels}
    
    return out


def get_filenames_labels(parent_dir):
    ''' String -> dict, dict, dict
    Returns three dictionaries containing all the filenames stored inside the parent directory. Each dictionary
    corresponds to one of the sets of the data (training, testing, validation).

    :param parent_dir: (String) Path to the parent directory.
    :return: Dictionaries containing the data with keys ['normal', 'altpig', 'dmae', 'excavation', 'membrana', 'nevus']
    '''

    pathologies = ['altpig', 'dmae', 'excavation', 'membrana', 'nevus']
    
    print('\n ---- Retreving TRAIN filenames ---- \n')
    train = {}
    train['normal'] = get_filenames_labels_mode_healthy(parent_dir=parent_dir, mode_to_retrieve='train')
    for pathology in pathologies:
        train[pathology] = get_filenames_labels_mode_pathologies(parent_dir=parent_dir,
                                                                 mode_to_retrieve='train',
                                                                 class_type_to_retrieve=pathology)
    print('\n ---- Retreving TEST filenames ---- \n')
    test = {}
    test['normal'] = get_filenames_labels_mode_healthy(parent_dir=parent_dir, mode_to_retrieve='test')
    for pathology in pathologies:
        test[pathology] = get_filenames_labels_mode_pathologies(parent_dir=parent_dir,
                                                                mode_to_retrieve='test',
                                                                class_type_to_retrieve=pathology)
    print('\n ---- Retreving VALIDATION filenames ---- \n')
    validation = {}
    validation['normal'] = get_filenames_labels_mode_healthy(parent_dir=parent_dir, mode_to_retrieve='validation')
    for pathology in pathologies:
        validation[pathology] = get_filenames_labels_mode_pathologies(parent_dir=parent_dir,
                                                                      mode_to_retrieve='validation',
                                                                      class_type_to_retrieve=pathology)
        
    return train, test, validation


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_list_to_TFRecord(image_paths, labels, output_dir, output_name, max_size = 0, resize=False, new_size=None):
    '''
    :param image_paths           List of file-paths for the images.
    :param labels                Class-labels for the images.
    :param output_name           File-name for the TFRecords output file, without the extension (.tfrecords)
    :param max_size              Maximum number of images to be placed in the created file. Unlimited if max_size = 0.
                                 If max_size is exceeded, more than one file will be created.
    :param max_size              Number of images saved in each .tfrecords file
    :param resize                Boolean indicating whether we want the images resized before saving them into TFRecords
    :param new_size              Integer indicating the new size of the images. Deprecated if resize = False
    '''

    assert len(image_paths) == len(labels), 'Number of image paths and labels do not coincide.'
    assert isinstance(max_size, int), 'Parameter max_size must be an integer'
    assert max_size >= 0, 'Parameter max_size must be equal or greater than 0'

    # General case, only one file needs to be created (max_size = 0)
    n_files_to_be_created = 1

    # Case in which max_size != 0
    if max_size != 0:
        # Check whether the list of image_path is greater than max_size
        if len(image_paths) > max_size:
            n_files_to_be_created = (len(image_paths) // max_size) + 1
            print('WARNING: Max_size is exceeded. {} files will be created with maximum size {} to place '
                  'the {} images and labels'.format(n_files_to_be_created, max_size, len(image_paths)))
            
    # Create the name of the file/s that will be created
    files_ids = ['balanced'] + [f for f in range(n_files_to_be_created)]  # Give a different name to the first file
    filenames_to_be_created = [output_name + '_' + str(f) + '.tfrecords' for f in files_ids]
            
    for file_id, filename_to_be_created in enumerate(filenames_to_be_created):
        print('----- Creating {} file -----'.format(filename_to_be_created))
        # Output file path
        output_path = os.path.join(output_dir, filename_to_be_created)
        
        # Choose which filenames and labels need to be placed in this file
        if n_files_to_be_created > 1:
            initial_idx = max_size * file_id
            final_idx = max_size * (file_id + 1)
            image_paths_batch = image_paths[initial_idx:final_idx]
            labels_batch = labels[initial_idx:final_idx]

        else:
            image_paths_batch = image_paths
            labels_batch = labels

        # Open a new TFRecordWriter to write the filenames and labels into a TFRecords file
        with tf.python_io.TFRecordWriter(output_path) as writer:
            # Iterate over all the image-paths and class-labels.
            for i, (path, label) in enumerate(zip(image_paths_batch, labels_batch)):

                # Print the progress
                if i % 200 == 0:
                    print('Progress: {}/{} images converted'.format(i, len(labels_batch)))
                    sys.stdout.flush()

                # Load the image-file using matplotlib's imread function.
                img = Image.open(path)
                # Optional: Resize the image to the desired size
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

        # Check that the created file has the expected amount of files
        print(output_path)
        len_tfrecord = len([x for x in tf.python_io.tf_record_iterator(output_path)])    
        assert len_tfrecord == len(image_paths_batch), 'Expected size of the created TFRecord file does ' \
                                                       'not coincide with the input size of the list of filenames'
    
    # Flush the memory
    sys.stdout.flush()


def create_TFRecords(train, test, validation, output_dir):

    print('\n ---- Creating TRAIN files in TFRecords format ----')
    for pathology in train.keys():
        convert_list_to_TFRecord(
            image_paths=train[pathology]['filenames'], 
            labels=train[pathology]['labels'],
            output_dir=output_dir,
            output_name='train/train_{}'.format(pathology),
            max_size=951)

    print('\n ---- Creating TEST files in TFRecords format ----')
    for pathology in test.keys():
        convert_list_to_TFRecord(
            image_paths=test[pathology]['filenames'], 
            labels=test[pathology]['labels'],
            output_dir=output_dir,
            output_name='test/test_{}'.format(pathology),
            max_size=298)

    print('\n ---- Creating VALIDATION files in TFRecords format ----')
    for pathology in validation.keys():
        convert_list_to_TFRecord(
            image_paths=validation[pathology]['filenames'], 
            labels=validation[pathology]['labels'],
            output_dir=output_dir,
            output_name='validation/validation_{}'.format(pathology),
            max_size=238)


if __name__ == '__main__':
    # Create an ArgumentParser object to be able to specify some parameters when executing the file
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=None,
                        help="Directory containing the original datasets")
    parser.add_argument('--output_dir', default=None,
                        help="Target directory where to save the files created")

    # Get the data_dir from the ArgParser object
    args = parser.parse_args()

    assert os.path.isdir(args.input_dir), 'The input directory does not exist'
    assert os.path.isdir(args.output_dir), 'The output directory does not exist'

    print('##### Getting the filenames and labels #### \n')
    train, test, validation = get_filenames_labels(parent_dir=args.input_dir)

    print('\n ##### Creating the TFRecords files #### \n')
    create_TFRecords(train, test, validation, output_dir=args.output_dir)
