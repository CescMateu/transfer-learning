"""Train the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.input_fn import input_fn
from model.utils import Params
from model.utils import set_logger
from model.model_fn import model_fn
from model.training import train_and_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/SIGNS_TFRecord_v2',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")


if __name__ == '__main__':
    # Set the random seed for the whole graph for being able to reproduce the experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    # assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Read the number of examples of each dataset (This could be optimized by storing the number in an external file
    # when creating the TFRecords)
    params.train_size = len([x for x in tf.python_io.tf_record_iterator(os.path.join(args.data_dir, 'train.tfrecords'))])
    logging.info('Train size: {}'.format(params.train_size))
    params.eval_size = len([x for x in tf.python_io.tf_record_iterator(os.path.join(args.data_dir, 'dev.tfrecords'))])
    logging.info('Test size: {}'.format(params.eval_size))

    # Create the input data pipeline
    logging.info("Reading the TFRecords training file...")
    train_inputs = input_fn(
        tfrecord_filenames=os.path.join(args.data_dir, 'train.tfrecords'),
        num_records=params.train_size,
        params=params
    )

    logging.info("Reading the TFRecords evaluation file...")
    eval_inputs = input_fn(
        tfrecord_filenames=os.path.join(args.data_dir, 'dev.tfrecords'),
        num_records=params.eval_size,
        params=params
    )

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn(train_inputs, params, is_training=True)
    eval_model_spec = model_fn(eval_inputs, params, is_training=False)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)