"""Tensorflow utility functions for evaluating the model"""

import logging
import tensorflow as tf

def evaluate_sess(sess, model_spec, num_steps):
    '''
    TODO: Finish documentation
    :param sess:
    :param model_spec:
    :param num_steps:
    :return:
    '''

    # Initialize the iterator with the dev dataset
    sess.run(model_spec['iterator'])

    # Iterate for one epoch and compute the accuracy for every batch in the dev set
    acc = 0
    for i in range(num_steps):
        acc += sess.run(model_spec['accuracy'])

    # Compute the mean accuracy over all the batches
    acc = acc/num_steps

    return acc


def evaluate(model_spec, params):
    '''
    Train the model and evaluate every epoch.

    :param dev_model_spec: (dict) contains the graph operations or nodes needed for evaluation. Contains
    [images, labels, iterator, variable_init_op, predictions, loss, accuracy]
    :param params: (Params) contains hyperparameters of the model.
    Must define: num_epochs, train_size, batch_size

    :return:
    '''

    with tf.Session() as sess:
        # Initialize model global variables
        sess.run(model_spec['variable_init_op'])

        # Compute the number of needed steps
        num_steps = (params.dev_size + params.batch_size - 1) // params.batch_size
        acc = evaluate_sess(sess, model_spec, num_steps)

        # Log the results
        logging.info('Evaluation accuracy with the dev set: {}'.format(acc))
