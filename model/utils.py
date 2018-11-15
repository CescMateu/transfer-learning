"""General utility functions"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import os


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def confusion_matrix(predictions_file, labels_file, params):
    '''
    Given the a predictions and labels file in .npy format, this function returns a np.array with the corresponding
    confusion matrix
    '''

    assert os.path.isfile(predictions_file), 'The predictions file provided does not exist'
    assert os.path.isfile(labels_file), 'The labels file provided does not exist'

    # Load the files
    preds = np.load(predictions_file)
    labs = np.load(labels_file)

    # Sanity check
    assert len(labs) == len(preds), 'The real and predicted vectors must have the' \
                                    ' same length: {} != {}'.format(len(labs), len(preds))
    n = len(labs)

    # Initialize the confusion matrix
    conf_matrix = np.zeros(shape=(params.num_labels, params.num_labels), dtype=np.int32)

    # Iterate over all the classes and all the elements in the vectors
    for idx in range(n):
        conf_matrix[labs[idx]][preds[idx]] += 1

    return np.around(conf_matrix, decimals=2)


def plot_and_save_confusion_matrix(cm, names, model_dir, mode, title='Confusion matrix',
                                   output_fname='confusion_matrix.jpg', cmap=plt.cm.Blues):
    '''
    Given a confusion matrix in np.array format (see confusion_matrix() from this same file), this function
    plots it and stores it in "model_dir"/plots/"mode"/"output_fname".
    '''

    # Define the output directory path
    plots_dir = os.path.join(model_dir, 'plots')
    plots_mode_dir = os.path.join(plots_dir, mode)

    # Create a directory where to save the plots if necessary
    if not os.path.isdir(plots_dir):
        os.system('mkdir {}'.format(plots_dir))

    if not os.path.isdir(plots_mode_dir):
        logging.info('Creating output directory for storing plots: {}'.format(plots_mode_dir))
        os.system('mkdir {}'.format(plots_mode_dir))

    # Start plotting the confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save the image in the created folder
    output_file = os.path.join(plots_mode_dir, output_fname)
    plt.savefig(output_file)
    plt.close()


def save_predictions_labels(sess, model_spec, num_steps, model_dir, mode):
    '''
    Saves the labels and predictions for the evaluation dataset in external files in order to be able to use them
    for computing assessment metrics (like the confusion matrix)
    '''

    assert mode in ['train', 'validation', 'test'], 'Parameter "mode" must be either "train", "evaluation" or "test"'

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['it_init_op'])
    sess.run(model_spec['metrics_init_op'])

    # Create the results directory if necessary
    results_dir = os.path.join(model_dir, 'results')
    if not os.path.isdir(results_dir):
        os.system('mkdir {}'.format(results_dir))

    results_eval_dir = os.path.join(results_dir, mode)
    if not os.path.isdir(results_eval_dir):
        logging.info('Creating the "{}" directory'.format(results_eval_dir))
        os.system('mkdir {}'.format(results_eval_dir))

    # Define the names of the files to be created
    predictions_file = os.path.join(results_eval_dir, 'predictions.npy')
    labels_file = os.path.join(results_eval_dir, 'labels.npy')

    first_run = True

    for _ in range(num_steps):
        # Run the labels and predictions for the following batch of images
        lab, pred = sess.run([model_spec['labels'], model_spec['predictions']])

        if first_run:
            # Predictions
            np.save(predictions_file, pred)
            # Labels
            np.save(labels_file, lab)
            # Change mode in the loop
            first_run = False

        else:
            # Predictions
            pred_file = np.load(predictions_file)
            np.save(predictions_file, np.concatenate((pred_file, pred)))
            # Labels
            lab_file = np.load(labels_file)
            np.save(labels_file, np.concatenate((lab_file, lab)))