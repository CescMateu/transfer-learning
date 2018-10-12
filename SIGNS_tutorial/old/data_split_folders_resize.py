import os
import random

from PIL import Image
from tqdm import tqdm

DATA_DIR = 'data/SIGNS'
OUTPUT_DATA_DIR = 'data/SIGNS_processed'
SIZE = 64


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


# Locate the training and test directories
train_dir = os.path.join(DATA_DIR, 'train_signs')
test_dir = os.path.join(DATA_DIR, 'test_signs')

# Get all the files from the directories
filenames = os.listdir(train_dir)
test_filenames = os.listdir(test_dir)

# Modify the paths to the files to be 'absolute'. (From the root of the project)
filenames = [os.path.join(train_dir, f) for f in filenames if f.endswith('jpg')]
test_filenames = [os.path.join(test_dir, f) for f in test_filenames if f.endswith('jpg')]

# Create a 'dev' dataset
# Make sure to always shuffle with a fixed seed so that the split is reproducible
random.seed(1234)
filenames.sort()
random.shuffle(filenames)
# Split the images in 'train_signs' into 80% train and 20% dev
split = int(0.8*len(filenames))
train_filenames = filenames[:split]
dev_filenames = filenames[split:]
# Put all the paths in a common dict
filenames = {'train': train_filenames, 'dev': dev_filenames, 'test': test_filenames}

# Preprocess train, dev and test
os.mkdir(OUTPUT_DATA_DIR)
for split in ['train', 'dev', 'test']:
    output_dir_split = os.path.join(OUTPUT_DATA_DIR, '{}_signs'.format(split))
    if not os.path.exists(output_dir_split):
        os.mkdir(output_dir_split)
        print('Creating folder ...')
    else:
        print("Warning: dir {} already exists".format(output_dir_split))

    print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))

    for filename in tqdm(filenames[split]):
        resize_and_save(filename, output_dir_split, size=SIZE)









#dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
#dataset = dataset.shuffle(len(filenames))
#dataset = dataset.map(parse_function, num_parallel_calls=4)
#dataset = dataset.map(train_preprocess, num_parallel_calls=4)
#dataset = dataset.batch(batch_size)
#dataset = dataset.prefetch(1)