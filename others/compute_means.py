import glob
import argparse
import os
import numpy as np
from PIL import Image
import random

SIZE = 512
N = 200

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/retina_data_subset',
                    help="Directory where all the images are contained")

parser.add_argument('--output_dir', default=None,
                    help="Directory where to save the results")

if __name__ == '__main__':

    # Load the input arguments
    args = parser.parse_args()

    # Sanity checks
    assert os.path.isdir(args.data_dir), 'Data directory does not exist'
    assert os.path.isdir(args.output_dir), 'Output directory does not exist'

    # Define the different pathologies and start extracting the images from the subfolders
    pathologies = ['altpig', 'dmae', 'excavation', 'membrana', 'nevus']
    partitions = ['train']

    filenames = []
    pattern = '*.jpg'
    print('--- Extracting images from the following directories: ')
    for pathology in pathologies:
        pathology_dir = 'u_{}_symbolic_512'.format(pathology)
        for partition in partitions:

            # if pathology == 'excavation':
            #     pathology = 'excavacion'
            for class_type in [pathology, 'normal']:
                fnames_dir = os.path.join(args.data_dir, pathology_dir, partition, class_type)
                assert os.path.isdir(fnames_dir), '{} directory does not exist'.format(fnames_dir)
                print('\t' + fnames_dir)
                filenames = filenames + glob.glob(os.path.join(args.data_dir,pathology_dir,partition,class_type,pattern))

    # Shuffle the images
    random.shuffle(filenames)
    print('--- {} images retrieved'.format(len(filenames)))

    # Retrieve N images (just for working in local)
    if N == 0:
        images = np.array([np.array(Image.open(filename)) for filename in filenames])
    else:
        images = np.array([np.array(Image.open(filename)) for filename in filenames[0:N]])

    print('--- Saving output files in {}'.format(args.output_dir))
    # Mean of each channel over all the images
    mean_channels = np.mean(images, axis=(0, 1, 2)).astype('uint8')
    mean_channels_img = np.zeros((SIZE, SIZE, 3), dtype='uint8') + mean_channels
    np.save(os.path.join(args.output_dir, 'mean_channels_img.npy'), mean_channels_img)

    # Mean of each pixel from each channel over all the images
    mean_img = np.mean(images, axis=0).astype('uint8')
    np.save(os.path.join(args.output_dir, 'mean_img.npy'), mean_img)
    print('Process finished successfully!')