import tensorflow as tf
import pickle
from utils.data_manipulation import create_sequences
from custom_cells.lstm import LSTM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor, KernelDensity
from collections import defaultdict
from PyNomaly.loop import LocalOutlierProbability
from scipy.integrate import quad
from scipy.special import erf

plt.style.use("seaborn")

def generate_pmf(x, n_bins):
    counts, bins = np.histogram(x, bins=n_bins)
    bins = bins[:-1] + (bins[1] - bins[0]) / 2
    probabilities = counts / np.sum(counts)
    print(bins)

def scale(x, low, high):
    return (np.array(x) - low) / (high - low)


def mean_of_dictionaries(*args):
    summation = defaultdict(list)
    for dictionary in args:
        for key in dictionary:
            summation[key].append(dictionary[key])

    for key in summation:
        summation[key] = np.array(summation[key]).mean(axis=0)

    return dict(summation)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_gates(look_back):
    prefix = "rnn/rnn/multi_rnn_cell/cell_0/lstm/"
    i = tf.get_default_graph().get_tensor_by_name(prefix + "input_gate_" + str(look_back-1) + ":0")
    f = tf.get_default_graph().get_tensor_by_name(prefix + "forget_gate_" + str(look_back-1) + ":0")
    o = tf.get_default_graph().get_tensor_by_name(prefix + "output_gate_" + str(look_back-1) + ":0")

    return i, f, o

def get_candidate(look_back):
    prefix = "rnn/rnn/multi_rnn_cell/cell_0/lstm/"
    c_ = tf.get_default_graph().get_tensor_by_name(prefix + "candidate_" + str(look_back - 1) + ":0")

    return c_

def train_and_eval(X_train, Y_train, X_test, Y_test, X_trigger, Y_trigger):
    X = tf.placeholder(dtype=tf.float64, shape=[None, lb, 1])
    Y = tf.placeholder(dtype=tf.float64, shape=[None, 1])

    network = tf.contrib.rnn.MultiRNNCell([LSTM(num_units=n_hidden, state_is_tuple=True)])
    hs, state_tuple = tf.nn.static_rnn(cell=network, inputs=tf.unstack(X, axis=1), dtype=tf.float64)
    c, h = state_tuple[0]

    with tf.variable_scope("params"):
        W = tf.get_variable("W", shape=[n_hidden, 1], dtype=tf.float64, initializer=tf.initializers.variance_scaling())
        b = tf.get_variable("b", shape=[1], dtype=tf.float64, initializer=tf.initializers.variance_scaling())

    preds = tf.matmul(h, W) + b
    cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=Y, predictions=preds))
    train_step = tf.train.AdamOptimizer(1e-1).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    n_iter = 100
    train_loss, test_loss = [], []
    for epoch in range(n_iter):
        _, train_mse = sess.run([train_step, cost], feed_dict={X: X_train, Y: Y_train})
        train_loss.append(train_mse)

        test_mse = sess.run(cost, feed_dict={X: X_test, Y: Y_test})
        test_loss.append(test_mse)

    weights, b = sess.run([W, b], feed_dict={X: X_trigger, Y: Y_trigger})

    i, f, o = get_gates(look_back=lb)
    c_ = get_candidate(look_back=lb)
    components = sess.run([h, c, c_, i, f, o], feed_dict={X: X_trigger, Y: Y_trigger})
    components = [component.T for component in components]

    hypothesis_test = sess.run(preds, feed_dict={X: X_test, Y: Y_test})

    tf.reset_default_graph()
    sess.close()

    return components, weights, b, hypothesis_test


if __name__ == "__main__":

    with open("datasets/sinusoids/data1.pkl", "rb") as f:
        data_dict = pickle.load(f)


    from sklearn.preprocessing import MinMaxScaler

    data_dict = dict()
    data_dict["train"] = MinMaxScaler().fit_transform(np.load("datasets/modifiedDatasets/sunspot/data5/train.npy").reshape(-1, 1)).ravel()
    data_dict["test"] = MinMaxScaler().fit_transform(np.load("datasets/modifiedDatasets/sunspot/data5/test.npy").reshape(-1, 1)).ravel()
    data_dict["trigger"] = MinMaxScaler().fit_transform(np.load("datasets/modifiedDatasets/sunspot/data5/outl.npy").reshape(-1, 1)).ravel()

    lb = 5
    n_hidden = 8

    X_train, Y_train = create_sequences(data_dict["train"], look_back=lb)
    X_test, Y_test = create_sequences(data_dict["test"], look_back=lb)
    X_trigger, Y_trigger = create_sequences(data_dict["trigger"], look_back=lb)

    list_of_lofs_per_run = []
    list_of_test_lofs_per_run = []
    for i in range(5):
        components, weights, b, hypothesis_test = train_and_eval(X_train, Y_train, X_test, Y_test, X_trigger, Y_trigger)
        lof = LocalOutlierFactor().fit(hypothesis_test).negative_outlier_factor_ * -1
        list_of_test_lofs_per_run.append(lof)


        lofs_dict = {}
        names = ["h", "c", "c_", "i", "f", "o", "ifo"]
        print(components[-1].shape)
        print(np.dstack((components[-1], components[-2], components[-3])).shape)
        components.append(np.dstack((components[-1], components[-2], components[-3])))

        for name, component in zip(names, components):
            lof_mat = np.zeros((component.shape[0], component.shape[1]))
            print(lof_mat.shape)
            for neuron, values in enumerate(component):
                #lof = LocalOutlierFactor().fit(values.reshape(-1, 1)).negative_outlier_factor_ * -1
                #lof = LocalOutlierProbability(values).fit().local_outlier_probabilities
                if name != "ifo":
                    lof = LocalOutlierFactor().fit(values.reshape(-1, 1)).negative_outlier_factor_ * -1
                else:
                    lof = LocalOutlierFactor().fit(values).negative_outlier_factor_ * -1

                #perplexity = .5
                #confidence = 1 - np.maximum(np.zeros_like(lof), erf((lof - np.mean(lof, axis=0)) / (2 * np.sqrt(2) * perplexity)))
                """confidence = 1 - np.maximum(np.zeros_like(lof), erf(
                    (lof - np.mean(lof)) / (np.std(lof) * 2 * np.sqrt(2))))"""
                lof_mat[neuron] = lof

            #overall_lof = np.mean(lof_mat, axis=0).ravel().tolist()
            #weights = np.abs(np.log(np.abs(weights)))
            w = weights / np.sum(np.abs(weights))
            overall_lof = (lof_mat.T.dot(1 - w)).ravel() / np.sum(1 - w).ravel()
            perplexity = np.linalg.norm(overall_lof)
            overall_lof = 1 - np.maximum(np.zeros_like(overall_lof), erf((overall_lof - np.mean(overall_lof)) / (perplexity)))
            lofs_dict[name] = overall_lof
        list_of_lofs_per_run.append(lofs_dict)

    results_dict = mean_of_dictionaries(*list_of_lofs_per_run)
    #mean_test_lof = np.mean(list_of_test_lofs_per_run, axis=0)
    #std_test_lof = np.std(list_of_test_lofs_per_run)

    #kde = KernelDensity(kernel="exponential", bandwidth=0.5).fit(mean_test_lof.reshape(-1, 1))

    #plt.plot(np.exp(kde.score_samples(results_dict["h"].reshape(-1, 1))))
    #plt.show()

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["h"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Hidden States")
    plt.show()

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["c"], linewidth=1)
    #axes[1].axhline(y=max_test_lof, linewidth=2, linestyle="--", color="grey")
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Internal States")
    plt.show()

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["c_"], linewidth=1)
    #axes[1].axhline(y=max_test_lof, linewidth=2, linestyle="--", color="grey")
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Candidate States")
    plt.show()

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["i"], linewidth=1)
    #axes[1].axhline(y=max_test_lof, linewidth=2, linestyle="--", color="grey")
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Input Gates")
    plt.show()

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["f"], linewidth=1)
    #axes[1].axhline(y=max_test_lof, linewidth=2, linestyle="--", color="grey")
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Forget Gates")
    plt.show()

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["o"], linewidth=1)
    #axes[1].axhline(y=max_test_lof, linewidth=2, linestyle="--", color="grey")
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Output Gates")
    plt.show()

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["ifo"], linewidth=1)
    # axes[1].axhline(y=max_test_lof, linewidth=2, linestyle="--", color="grey")
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Input/Forget/Output Gates")
    plt.show()




    #exit()





    X = tf.placeholder(dtype=tf.float64, shape=[None, lb, 1])
    Y = tf.placeholder(dtype=tf.float64, shape=[None, 1])

    network = tf.contrib.rnn.MultiRNNCell([LSTM(num_units=n_hidden, state_is_tuple=True)])
    hs, state_tuple = tf.nn.static_rnn(cell=network, inputs=tf.unstack(X, axis=1), dtype=tf.float64)
    c, h = state_tuple[0]

    with tf.variable_scope("params"):
        W = tf.get_variable("W", shape=[n_hidden, 1], dtype=tf.float64, initializer=tf.initializers.variance_scaling())
        b = tf.get_variable("b", shape=[1], dtype=tf.float64, initializer=tf.initializers.variance_scaling())
        #W = tf.Variable(initial_value=np.random.randn(n_hidden, 1), dtype=tf.float64)
        #b = tf.Variable(initial_value=np.random.randn(1), dtype=tf.float64)


    # hidden state of each neuron per timestep
    #print(hs)
    # last internal state
    #print(c)
    # last hidden state
    #print(h)

    preds = tf.matmul(h, W) + b

    cost = tf.reduce_mean(tf.losses.mean_squared_error(labels=Y, predictions=preds))
    train_step = tf.train.AdamOptimizer(1e-1).minimize(cost)
    #train_step = tf.train.GradientDescentOptimizer(1e-1).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    n_iter = 100
    train_loss, test_loss = [], []
    for epoch in range(n_iter):
        _, train_mse = sess.run([train_step, cost], feed_dict={X: X_train, Y: Y_train})
        train_loss.append(train_mse)

        test_mse = sess.run(cost, feed_dict={X: X_test, Y: Y_test})
        test_loss.append(test_mse)

    plt.plot(train_loss, label="Train")
    plt.plot(test_loss, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

    hypothesis_clean = sess.run(preds, feed_dict={X: X_test, Y: Y_test})
    plt.plot(hypothesis_clean, label="Hypothesis")
    plt.plot(Y_test, label="Groundtruth")
    plt.legend()
    plt.show()

    hypothesis_trigger = sess.run(preds, feed_dict={X: X_trigger, Y: Y_trigger})
    plt.plot(Y_trigger, label="Groundtruth")
    plt.plot(hypothesis_trigger, label="Hypothesis")
    plt.legend()
    plt.show()

    # RNN forecast error
    error = np.abs(hypothesis_trigger - Y_trigger)
    plt.plot(error)
    plt.show()

    # get weights W
    weights, b = sess.run([W, b], feed_dict={X: X_trigger, Y: Y_trigger})

    # hidden states
    hidden_state = sess.run(h, feed_dict={X: X_trigger, Y: Y_trigger}).T
    #hidden_state = np.tanh(hidden_state)
    lof_mat = np.zeros_like(hidden_state)
    for neuron, h in enumerate(hidden_state):
        lof = LocalOutlierFactor().fit(h.reshape(-1, 1)).negative_outlier_factor_ * -1
        lof_mat[neuron] = lof

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=2)
    for h in hidden_state:
        axes[1].plot(h, linewidth=1)
    axes[2].plot(np.mean(lof_mat, axis=0))
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Hidden States")
    axes[2].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Hidden States")
    plt.show()

    # internal states
    internal_state = sess.run(c, feed_dict={X: X_trigger, Y: Y_trigger}).T
    #internal_state = sigmoid(internal_state)
    lof_mat = np.zeros_like(internal_state)
    for neuron, c in enumerate(internal_state):
        lof = LocalOutlierFactor().fit(c.reshape(-1, 1)).negative_outlier_factor_ * -1
        lof_mat[neuron] = lof

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=2)
    for c in internal_state:
        axes[1].plot(c, linewidth=1)
    axes[2].plot(lof_mat.T.dot(weights))
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Hidden States")
    axes[2].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Internal States")
    plt.show()

    i, f, o = get_gates(look_back=lb)
    c_ = get_candidate(look_back=lb)
    components = sess.run([i, f, o, c_], feed_dict={X: X_trigger, Y: Y_trigger})
    input_gate, forget_gate, output_gate, candidate = [component.T for component in components]
    #input_gate = sigmoid(input_gate)
    #forget_gate = sigmoid(forget_gate)
    #output_gate = sigmoid(output_gate)
    #candidate = np.tanh(candidate)

    # candidate
    lof_mat = np.zeros_like(candidate)
    for neuron, c_ in enumerate(candidate):
        lof = LocalOutlierFactor().fit(c_.reshape(-1, 1)).negative_outlier_factor_ * -1
        lof_mat[neuron] = lof

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=2)
    for c_ in candidate:
        axes[1].plot(c_, linewidth=1)
    axes[2].plot(lof_mat.T.dot(weights))
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Candidate States")
    axes[2].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.show()

    # input gates
    lof_mat = np.zeros_like(input_gate)
    for neuron, i in enumerate(input_gate):
        lof = LocalOutlierFactor().fit(i.reshape(-1, 1)).negative_outlier_factor_ * -1
        lof_mat[neuron] = lof

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=2)
    for i in input_gate:
        axes[1].plot(i, linewidth=1)
    axes[2].plot(lof_mat.T.dot(weights))
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Input Gates")
    axes[2].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.show()

    # output gates
    lof_mat = np.zeros_like(output_gate)
    for neuron, o in enumerate(output_gate):
        lof = LocalOutlierFactor().fit(o.reshape(-1, 1)).negative_outlier_factor_ * -1
        lof_mat[neuron] = lof

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=2)
    for o in output_gate:
        axes[1].plot(o, linewidth=1)
    axes[2].plot(lof_mat.T.dot(weights))
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Output Gates")
    axes[2].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.show()

    # forget gates
    lof_mat = np.zeros_like(forget_gate)
    for neuron, f in enumerate(forget_gate):
        lof = LocalOutlierFactor().fit(f.reshape(-1, 1)).negative_outlier_factor_ * -1
        lof_mat[neuron] = lof

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=2)
    for f in forget_gate:
        axes[1].plot(f, linewidth=1)
    axes[2].plot(lof_mat.T.dot(weights))
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Forget Gates")
    axes[2].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.show()




