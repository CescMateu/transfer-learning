from cnn import CNN


# Input pipeline (Aquí es defineixen els tensors d'entrada: La imatge, la seva label corresponent i en aquest cas el ràtio de dropout que es desitja)
input_x = tf.placeholder(tf.float32, shape=(None, 512, 512, 3))
labels = tf.placeholder(tf.int64, shape=(None, 1))

drop_rate = tf.placeholder(tf.float32)

# Model (Aquí connectem els tensors d'entrada a l'entrada de la arquitectura definida en un arxiu apart)
with tf.name_scope("model"):
    cnn = cnn_v1(n_outputs=n_labels, dropout_rate=drop_rate)
    cnn.build(input_x)
    logits = cnn.logits

# Model toppings (Aquí fem ús de la sortida de la arquitectura, és a dir, els logits)
predictions = tf.argmax(logits, axis=-1, name="predictions")
batch_loss = tf.losses.softmax_cross_entropy(
    onehot_labels=tf.one_hot(tf.squeeze(labels), depth=n_labels),
    logits=logits)
global_step = tf.Variable(0, trainable=False, name="global_step")
adam = tf.train.AdamOptimizer(learning_rate=parameters['learning_rate'])
train_op = adam.minimize(batch_loss,
                         global_step=global_step,
                         name="train_op")