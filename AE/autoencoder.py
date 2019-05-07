from __future__ import division, print_function, absolute_import

import tensorflow as tf
import os
import pickle
from datetime import datetime as dt
from numpy import random

# import matplotlib.pyplot as plt

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 10,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_input", 16, "Number of features in training data.")
tf.app.flags.DEFINE_integer("num_hidden_units", 16, "Size of each hidden layer.")
tf.app.flags.DEFINE_integer("num_bottleneck_units", 6, "Size of bottleneck layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the encoder and decoder.")
tf.app.flags.DEFINE_string("train_dir", "AE_models/numLayers2_numHiddenUnits16_bottelneckUnits6_learningRate0.1", "Training directory.")
tf.app.flags.DEFINE_integer("num_epochs", 10,
                            "Maximium number of epochs for trainig.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 10,
                            "How many training steps to do per checkpoint.")

FLAGS = tf.app.flags.FLAGS


def _check_dir(dir):
    """
    Checks if a directory exists; if not, creates a new directory and goes on.
     if it does exist; checks if it contains a trained model. If that is the case, print e message a quit.
     If it doesn't contain a trained model, checks if the parameters match the current ones
    Raise: ValueError if dir contains a model with different parameters
            IOError if dir contains a model fully trained
    """

    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        if not os.path.isfile(dir + "/params.pickle"):
            _save_parameters(True)
        else:
            if _save_parameters(False) != pickle.load(open(dir + "/params.pickle", 'rb')):
                raise ValueError("%s  directory contains a model trained with different parameters" % (dir))
        checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if checkpoint and os.path.isfile(dir + "/results.pkl"):
            raise IOError("A model trained with these parameters already exists!")


def _save_parameters(save):
    params_dic = {}

    params_dic["batch_size"] = FLAGS.batch_size
    params_dic["num_layers"] = FLAGS.num_layers
    params_dic["num_hidden_units"] = FLAGS.num_hidden_units
    params_dic["num_bottleneck_units"] = FLAGS.num_bottleneck_units
    params_dic["num_input"] = FLAGS.num_input
    params_dic["learning_rate"] = FLAGS.learning_rate
    params_dic["num_epochs"] = FLAGS.num_epochs
    params_dic["steps_per_checkpoint"] = FLAGS.steps_per_checkpoint

    if save:
        pickle.dump(params_dic, open(FLAGS.train_dir + '/params.pickle', 'wb'))
    return params_dic


# tf Graph input
def graph_input(num_input):
    return tf.placeholder("float", [None, num_input])


def trainable_variables(num_input, num_hidden, num_bottleneck, num_layer):
    weights = {}
    biases = {}
    # for now, all hidden layers have the same number of units (except the bottleneck)
    current_input = num_input
    for h in range(num_layer - 1):
        weights['encoder_h' + str(h + 1)] = tf.Variable(tf.random_normal([current_input, num_hidden]))
        weights['decoder_h' + str(num_layer - h)] = tf.Variable(tf.random_normal([num_hidden, current_input]))
        biases['encoder_b' + str(h + 1)] = tf.Variable(tf.random_normal([num_hidden]))
        biases['decoder_b' + str(num_layer - h)] = tf.Variable(tf.random_normal([current_input]))
        current_input = num_hidden

    weights['encoder_h' + str(num_layer)] = tf.Variable(tf.random_normal([current_input, num_bottleneck]))
    weights['decoder_h' + str(1)] = tf.Variable(tf.random_normal([num_bottleneck, current_input]))
    biases['encoder_b' + str(num_layer)] = tf.Variable(tf.random_normal([num_bottleneck]))
    biases['decoder_b' + str(1)] = tf.Variable(tf.random_normal([current_input]))

    return weights, biases


# Building the autoencoder
def autoencoder(num_input, num_hidden, num_bottleneck, num_layers):
    # change the number of layers

    X = graph_input(num_input)
    weights, biases = trainable_variables(num_input, num_hidden, num_bottleneck, num_layers)

    current_input = X
    layers = {}
    for layer in range(num_layers):
        layers['encoder_layer_' + str(layer + 1)] = tf.nn.sigmoid(
            tf.add(tf.matmul(current_input, weights['encoder_h' + str(layer + 1)]), biases['encoder_b' + str(layer + 1)]))
        current_input = layers['encoder_layer_' + str(layer + 1)]

    encoding = layers['encoder_layer_' + str(num_layers)]

    for layer in range(num_layers):
        layers['decoder_layer_' + str(layer + 1)] = tf.nn.sigmoid(
            tf.add(tf.matmul(current_input, weights['decoder_h' + str(layer + 1)]), biases['decoder_b' + str(layer + 1)]))
        current_input = layers['decoder_layer_' + str(layer + 1)]

    reconstruction = layers['decoder_layer_' + str(num_layers)]

    return X, encoding, reconstruction


def next_batch(X, batch_size, i):
    start = batch_size * (i - 1)
    end = batch_size * (i - 1) + batch_size
    if end >= X.shape[0]:
        end = -1
    # batch_x = X.iloc[start: end].values
    batch_x = X[start: end, :]
    return batch_x


def train_model(X_features):
    X, encoder_op, decoder_op = autoencoder(FLAGS.num_input, FLAGS.num_hidden_units, FLAGS.num_bottleneck_units
                                            , FLAGS.num_layers)

    num_steps = int(len(X_features) / FLAGS.batch_size)
    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # Define loss and optimizer, minimize the squared error
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate).minimize(loss)

    # Start Training

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    # Start a new TF session
    with tf.Session(config=config) as sess:
        with tf.variable_scope("model") as scope:
            _check_dir(FLAGS.train_dir)
            result_file = open(FLAGS.train_dir + "/results.txt", 'a+')

            checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if checkpoint and tf.gfile.Exists(checkpoint.model_checkpoint_path):
                print("Reading model parameters from %s" % checkpoint.model_checkpoint_path)
                saver.restore(sess, checkpoint.model_checkpoint_path)
            else:
                print("Created model with fresh parameters.")
                sess.run(tf.global_variables_initializer())

            checkpoint_path = FLAGS.train_dir + "/autoencoder.ckpt"
            scope.reuse_variables()

            result_file.write("\n")
            result_file.write(str(dt.now()))
            result_file.write("\n")
            result_file.write(
                " %d hidden layers of %d units  %d embedding size %d bach-size. \n" %
                (FLAGS.num_layers, FLAGS.num_hidden_units, FLAGS.num_bottleneck_units, FLAGS.batch_size))
            result_file.write(
                " %d batch size %d number of steps to complete one epoch \n" % (FLAGS.batch_size, num_steps))

            result_dic = {}
            all_batch_losses = []
            all_epoch_losses = []
            # Training
            for i in range(1, FLAGS.num_epochs + 1):
                # shuffle data
                # X_curr = X_features.sample(frac=1).reset_index(drop=True)
                random.shuffle(X_features)
                for j in range(1, num_steps + 1):
                    # Prepare Data
                    # Get the next batch of data
                    batch_x = next_batch(X_features, FLAGS.batch_size, j)

                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, batch_loss = sess.run([optimizer, loss], feed_dict={X: batch_x})
                    all_batch_losses.append(batch_loss)
                    # Display logs per step and save current model
                    if j % FLAGS.steps_per_checkpoint == 0 or j == 1:
                        result_file.write('Epoch %i, Step %i: Minibatch Loss: %f  \n' % (i, j, batch_loss))
                        saver.save(sess, checkpoint_path)
                epoch_loss = sess.run(loss, feed_dict={X: X_features})
                all_epoch_losses.append(epoch_loss)
                result_file.write('Epoch %i finished, global Loss: %f  \n' % (i, epoch_loss))

            result_dic["epoch_losses"] = all_epoch_losses
            result_dic["batch_losses"] = all_batch_losses
            pickle.dump(result_dic, open(FLAGS.train_dir + '/results.pkl', 'wb'))
            result_file.close()
            # r = sess.run(encoder_op, feed_dict={X: X_features.values})
            # g = sess.run(decoder_op, feed_dict={encoder_op: r})

            # save the trained model
            saver.save(sess, checkpoint_path)


def main(_):
    X_features = random.rand(100, 16)
    train_model(X_features)


if __name__ == "__main__":
    tf.app.run()
