import tensorflow as tf
import numpy as np



inputDim = 1
seq_length = 2
RNNLayers = 1
outputDim = 1

model = tf.Graph()
with model.as_default():
    trainX = tf.placeholder(tf.float32, shape=[None, seq_length, inputDim], name="trainX")
    # trainY = tf.placeholder("float32", [outputDim], name="trainY")

    with tf.variable_scope("LSTM_layer"):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=inputDim, state_is_tuple=True, activation=tf.nn.tanh)
        cells = tf.nn.rnn_cell.MultiRNNCell([cell] * RNNLayers, state_is_tuple=True)
        output, _states = tf.nn.dynamic_rnn(cells, trainX, dtype=tf.float32)

    with tf.variable_scope("Fully_connected_layer"):
        X_FC = tf.contrib.layers.fully_connected(output, 3, activation_fn=None)


    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs", model)