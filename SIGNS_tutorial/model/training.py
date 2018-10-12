"""Tensorflow utility functions for training"""

import logging
import tensorflow as tf
import os

from evaluation import evaluate_sess

def get_weights():

    # CONV1 -> MAXPOOL1 -> CONV2 -> MAXPOOL2 -> FC1 -> FC2
    w_op = {
        'W_conv1': tf.trainable_variables()[0],
        'b_conv1': tf.trainable_variables()[1],
        'W_conv2': tf.trainable_variables()[2],
        'b_conv2': tf.trainable_variables()[3],
        'W_fc1': tf.trainable_variables()[4],
        'b_fc1': tf.trainable_variables()[5],
        'W_fc2': tf.trainable_variables()[6],
        'b_fc2': tf.trainable_variables()[7]
        }

    return w_op

def train_sess(sess, model_spec, num_steps):
    '''
    Train the model on `num_steps` batches

    :param sess: (tf.Session) current session
    :param model_spec: (dict) contains the graph operations or nodes needed for training
    :param num_steps: (int) train for this number of batches
    :param params: (Params) hyperparameters
    :return:
    '''

    # Extract the different parts of model_spec
    loss = model_spec['loss']
    train_op = model_spec['train_op']
    accuracy = model_spec['accuracy']

    # Load the training dataset into the pipeline
    sess.run(model_spec['iterator'])

    # Iterate for the current batch
    for i in range(num_steps):
        _, loss_val = sess.run([train_op, loss])
        if i % 10 == 0:
            acc = sess.run(accuracy)
            logging.info('Training accuracy at batch number {}: {}'.format(i, acc))


def train_and_evaluate(train_model_spec, eval_model_spec, params):
    '''
    Train the model and evaluate every epoch.

    :param model_spec: (dict) contains the graph operations or nodes needed for training
    [images, labels, iterator, variable_init_op, predictions, loss, accuracy, train_op]
    :param params: (Params) contains hyperparameters of the model.
    Must define: num_epochs, train_size, batch_size

    :return:
    '''

    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver()  # will keep last 5 epochs
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)

    with tf.Session() as sess:
        # Initialize model global variables
        sess.run(train_model_spec['variable_init_op'])

        best_dev_acc = 0
        for epoch in range(params.num_epochs):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

            # Compute number of batches in one epoch (one full pass over the training set)
            num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
            train_sess(sess, train_model_spec, num_steps)

            # Save weights
            last_save_path = os.path.join(params.model_dir, 'last_weights', 'after-epoch')
            last_saver.save(sess, last_save_path, global_step=epoch + 1)

            # Evaluate for one epoch on validation set
            num_steps = (params.dev_size + params.batch_size - 1) // params.batch_size
            dev_acc = evaluate_sess(sess, eval_model_spec, num_steps)

            # If best_eval, best_save_path
            if dev_acc >= best_dev_acc:
                # Store new best accuracy
                best_dev_acc = dev_acc

                # Save weights
                best_save_path = os.path.join(params.model_dir, 'best_weights', 'after-epoch')
                best_save_path = best_saver.save(sess, best_save_path, global_step=epoch + 1)
                logging.info("- Found new best accuracy of {:.3f}, saving in {}".format(best_dev_acc, best_save_path))