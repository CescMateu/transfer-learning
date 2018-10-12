import os
import logging
import tensorflow as tf
import sys

sys.path.insert(0, 'model/') # TODO: Fix this. Needed for importing the following scripts.
from utils import Params
from utils import set_logger
from input_fn import input_fn
from model_fn import model_fn
from training import train_and_evaluate

# -----------------------------------------------------------
# INITIALIZE PROGRAM

# Set the random seed for the whole graph for reproducible experiments
tf.set_random_seed(12345)

# Load the parameters for the experiment
json_path = os.path.join('experiments/params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = Params(json_path)

# Set the logger
set_logger(os.path.join(params.results_dir, 'train.log'))

# -----------------------------------------------------------
# LOAD DATA AND CREATE PIPELINE WITH TF.DATA

# Load the training filenames and labels
logging.info('Creating the training dataset...')
full_train_dir = os.path.join(params.data_dir, params.train_dir)
train_filenames = os.listdir(full_train_dir)
train_labels = [int(f[0]) for f in train_filenames if f.endswith('.jpg')]
train_filenames = [os.path.join(full_train_dir, f) for f in train_filenames if f.endswith('.jpg')]
params.train_size = len(train_filenames)

# Run input_fn and collect the dict with the training inputs
train_inputs = input_fn(train_filenames, train_labels, params, is_training=True)
logging.info('Train dataset created containing {} examples'.format(params.train_size))

# Load the developement filenames and labels
logging.info('Creating the dev dataset...')
full_dev_dir = os.path.join(params.data_dir, params.dev_dir)
dev_filenames = os.listdir(full_dev_dir)
dev_labels = [int(f[0]) for f in dev_filenames if f.endswith('.jpg')]
dev_filenames = [os.path.join(full_dev_dir, f) for f in dev_filenames if f.endswith('.jpg')]
params.dev_size = len(dev_filenames)

# Run input_fn and collect the dict with the dev inputs
dev_inputs = input_fn(dev_filenames, dev_labels, params, is_training=False)
logging.info('Dev dataset created containing {} examples'.format(params.dev_size))

# -----------------------------------------------------------
# DEFINE THE MODEL GRAPH

logging.info('Defining the model graph...')
train_model_spec = model_fn(train_inputs, params, is_training=True)
dev_model_spec = model_fn(dev_inputs, params, is_training=False, reuse=True)

# -----------------------------------------------------------
# TRAIN THE MODEL

logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
w_train = train_and_evaluate(train_model_spec, dev_model_spec, params)
logging.info("Finished training for {} epoch(s)".format(params.num_epochs))


