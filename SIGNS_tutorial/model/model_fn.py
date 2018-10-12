"""Define the model."""

import tensorflow as tf

from cnn import CNN

def build_model(is_training, inputs, params):
    '''
    TODO: Finish documentation

    Creates the corresponding TensorFlow graph that needs to be executed inside a tf.Session().

    :param input_x
    :param labels
    :param dropout_rate
    :param learning_rate

    :return: Two nodes of the tf.graph corresponding to the cost and the optimizer.
    '''

    # Separate the inputs objects
    images = inputs['images']

    # Feeding the 'inputs' data to the model
    cnn = CNN(params.dropout_rate)
    cnn.build(images)
    logits = cnn.logits

    return logits


def model_fn(inputs, params, is_training, reuse=False):
    '''
    TODO: Finish documentation

    :param inputs:
    :param params:
    :param is_training:
    :return:
    '''

    # Extract the inputs and cast them into the appropriate data types
    labels = tf.cast(inputs['labels'], tf.int64)
    images = tf.cast(inputs['images'], tf.float64)

    # -----------------------------------------------------------
    # MODEL CREATION
    # Define the layers of the model
    logits = build_model(inputs, params, reuse)
    predictions = tf.argmax(logits, 1)

    # Define loss and accuracy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        # TODO: Here we could add batch normalization
        train_op = optimizer.minimize(loss, global_step=global_step)

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)

    # TODO: Finish this part

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec


