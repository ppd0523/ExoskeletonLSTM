import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
# import fileReader as fr

inputDim = 3
cellSize = 5
fullyconnectedDim = 10
seq_length = 2
RNNLayers = 1
outputDim = 1
batchSize = 1

CHECK_POINT_DIR = "testModel2" # directory to save model states



# raw = np.loadtxt("./data/data0.txt", delimiter=',')
# row = raw.shape[0]
# col = raw.shape[1] #col=3
# trainSize = row-seq_length
trainSize = 1

# x, y = [], []
# for i in range(trainSize):
#     x.append(raw[i:i+seq_length])
#     y.append([raw[i+seq_length][-1]])

X = tf.placeholder(dtype=tf.float32, shape=[None, seq_length, inputDim], name="trainX")
Y = tf.placeholder(dtype=tf.float32, shape=[None, outputDim], name="trainY")
keep_prob = tf.placeholder(tf.float32, name="drop_out")

with tf.name_scope("LSTM_layer"):
    cell = tf.nn.rnn_cell.LSTMCell(num_units=cellSize, state_is_tuple=True, activation=tf.nn.tanh)
    # cells = tf.nn.rnn_cell.MultiRNNCell([cell] * RNNLayers, state_is_tuple=True)
    LSTMoutputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    tf.summary.histogram("LSTMoutput", LSTMoutputs)
    tf.summary.histogram("_states", _states)

with tf.name_scope("Fully_connected_layer0"):
    X_FC = tf.contrib.layers.fully_connected(LSTMoutputs, fullyconnectedDim, activation_fn=tf.nn.relu)
    X_FC = tf.contrib.layers.flatten(X_FC)

    tf.summary.histogram("X_FC", X_FC)


with tf.name_scope("Fully_connected_layer1"):
    W1 = tf.get_variable(shape=[seq_length*fullyconnectedDim, outputDim], name="W1")
    B1 = tf.get_variable(shape=[outputDim], name="B1")
    H1 = tf.add(tf.matmul(X_FC, W1), B1, name="H1")
    # H1 = tf.nn.relu(H1)
    Y_ = H1

    tf.summary.histogram("H1", H1)


merged = tf.summary.merge_all()

# real data
targets = tf.placeholder(dtype=tf.float32, shape=[None,outputDim], name="targets")

#
loss = tf.reduce_mean(tf.squared_difference(H1, Y))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=1, rho=0.95, epsilon=1e-8).minimize(loss)


total = trainSize//batchSize

start_epoch = tf.Variable(0, name="start_epoch")
last_epoch = tf.Variable(0, name="last_epoch")

global_step = tf.Variable(0, trainable=False)


with tf.Session() as sess:

    sess.run( tf.global_variables_initializer() )
    writer = tf.summary.FileWriter("./logs")

    saver = tf.train.Saver()
    ######### load ########
    # checkPoint = tf.train.get_checkpoint_state("./model/"+CHECK_POINT_DIR)
    # print(checkPoint, "\n\n",checkPoint.model_checkpoint_path)
    # if checkPoint and checkPoint.model_checkpoint_path:
    #     try:
    #         saver.restore(sess, checkPoint.model_checkpoint_path)
    #         print("loading")
    #     except:
    #         print('loading error')
    # else:
    #     print("no saved model")
    #######################
    start = sess.run(last_epoch)
    sess.run(last_epoch.assign(start + 1))

    ######### save ########
    MODEL_PATH = "./model/"+CHECK_POINT_DIR
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    saver.save(sess, MODEL_PATH+"/state", global_step=0)
    #######################


    # saver.save(sess, "./saved/")




    # for i in range(trainSize):
    #     _, summary = sess.run([optimizer, merged], feed_dict={X:x, Y:y})
    #
    #
    #     writer.add_summary(summary, 2)

    print(sess.run(start_epoch))
    print(sess.run(last_epoch))
    # test_predict = sess.run(Y_, feed_dict={X:testX})
    #
    # plt.plot()