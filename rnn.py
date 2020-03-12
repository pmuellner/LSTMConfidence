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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats.stats import pearsonr
from NPEET.npeet.entropy_estimators import mi, kldiv, shuffle_test

plt.style.use("seaborn")

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def average_precision(y_true, y_pred):
    precisions = []
    for k in range(len(y_true)):
        p_k = precision_score(y_true, y_pred)
        precisions.append(p_k)

    return np.mean(precisions)

def evaluate_classification(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    ap = average_precision(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    return {"Precision": precision, "Recall": recall, "F1-score": f1, "Average Precision": ap, "Area Under (ROC) Curve": auc}


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

    H_trigger = sess.run(preds, feed_dict={X: X_trigger, Y: Y_trigger})
    H_test = sess.run(preds, feed_dict={X: X_test, Y: Y_test})

    tf.reset_default_graph()
    sess.close()

    return components, weights, b, H_trigger, H_test


if __name__ == "__main__":

    with open("datasets/sinusoids/data1.pkl", "rb") as f:
        data_dict = pickle.load(f)



    from sklearn.preprocessing import MinMaxScaler
    data_dict = dict()
    data_dict["train"] = MinMaxScaler().fit_transform(np.load("datasets/modifiedDatasets/sunspot/data5/train.npy").reshape(-1, 1)).ravel()
    data_dict["test"] = MinMaxScaler().fit_transform(np.load("datasets/modifiedDatasets/sunspot/data5/test.npy").reshape(-1, 1)).ravel()
    data_dict["trigger"] = MinMaxScaler().fit_transform(np.load("datasets/modifiedDatasets/sunspot/data5/outl.npy").reshape(-1, 1)).ravel()
    data_dict["labels"] = np.load("datasets/modifiedDatasets/sunspot/data5/labels.npy")


    lb = 5
    n_hidden = 10

    X_train, Y_train = create_sequences(data_dict["train"], look_back=lb)
    X_test, Y_test = create_sequences(data_dict["test"], look_back=lb)
    X_trigger, Y_trigger = create_sequences(data_dict["trigger"], look_back=lb)
    labels = data_dict["labels"][lb+1:]

    list_of_lofs_per_run = []
    list_of_test_lofs_per_run = []
    list_of_kls_per_run = []
    for i in range(15):
        components, weights, b, hypothesis, hypothesis_test = train_and_eval(X_train, Y_train, X_test, Y_test, X_trigger, Y_trigger)
        #components, weights, b, hypothesis, hypothesis_test = train_and_eval(X_train, Y_train, X_test, Y_test, X_test, Y_test)
        lof_test = LocalOutlierFactor().fit(hypothesis_test).negative_outlier_factor_ * -1
        #list_of_test_lofs_per_run.append(lof)


        lofs_dict = {}
        kls_dict = {}
        names = ["h", "c", "c_", "i", "f", "o", "ifo"]
        components.append(np.dstack((components[-1], components[-2], components[-3])))

        for name, component in zip(names, components):
            lof_mat = np.zeros((component.shape[0], component.shape[1]))
            for neuron, values in enumerate(component):
                #lof = LocalOutlierProbability(values).fit().local_outlier_probabilities
                if name != "ifo":
                    lof = LocalOutlierFactor(n_neighbors=15, algorithm="brute").fit(values.reshape(-1, 1)).negative_outlier_factor_ * -1
                else:
                    lof = LocalOutlierFactor(n_neighbors=15, algorithm="brute").fit(values).negative_outlier_factor_ * -1

                lof_mat[neuron] = lof

            # TODO test if w or 1/w is better --> later
            w = np.abs(weights) / np.sum(np.abs(weights))
            #overall_lof = lof_mat.T.dot(1 - w).ravel()
            #overall_lof = lof_mat.T.dot(w).ravel()

            # construct pmf from overal lof
            #print(KL(np.random.randn(len(overall_lof)), overall_lof))

            overall_lof = np.mean(lof_mat, axis=0)
            kl = shuffle_test(kldiv, overall_lof.reshape(-1, 1), lof_test.reshape(-1, 1).reshape(-1, 1))[0]
            kls_dict[name] = kl

            #overall_lof = lof_mat.T.dot(weights)
            perplexity = 15
            #overall_lof = 1 - np.maximum(np.zeros_like(overall_lof), erf((overall_lof - np.mean(overall_lof)) / (np.sqrt(2) * np.std(overall_lof) * perplexity)))
            overall_lof = 1 - np.maximum(np.zeros_like(overall_lof), erf((overall_lof - np.mean(overall_lof)) / (np.sqrt(2) * kl * perplexity)))
            lofs_dict[name] = overall_lof
        list_of_lofs_per_run.append(lofs_dict)
        list_of_kls_per_run.append(kls_dict)

    results_dict = mean_of_dictionaries(*list_of_lofs_per_run)
    kl_dict = mean_of_dictionaries(*list_of_kls_per_run)
    print(kl_dict)

    variances = []
    n_neighbors = list(np.arange(10, 50, 1))
    for k in n_neighbors:
        lof = LocalOutlierFactor(n_neighbors=k, algorithm="brute").fit(np.array(data_dict["test"]).reshape(-1, 1)).negative_outlier_factor_ * -1
        variances.append(np.var(lof))
    plt.plot(n_neighbors, variances)
    plt.show()

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(LocalOutlierFactor().fit(np.array(data_dict["trigger"]).reshape(-1, 1)).negative_outlier_factor_ * -1, linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("X")
    plt.show()

    print(evaluate_classification(y_true=labels, y_pred=(results_dict["h"] < 0.5)))
    print("Mutual Information component-trigger: %f" % shuffle_test(mi, results_dict["h"], Y_trigger.ravel(), alpha=0.25)[0])
    print("Mutual Information component-hypothesis: %f" % shuffle_test(mi, results_dict["h"], hypothesis.ravel(), alpha=0.25)[0])
    print("Mutual Information component-error: %f" % shuffle_test(mi, results_dict["h"], np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0])
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["h"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Hidden States")
    plt.show()

    print(evaluate_classification(y_true=labels, y_pred=(results_dict["c"] < 0.5)))
    print("Mutual Information component-trigger: %f" % shuffle_test(mi, results_dict["c"], Y_trigger.ravel(), alpha=0.25)[0])
    print("Mutual Information component-hypothesis: %f" % shuffle_test(mi, results_dict["c"], hypothesis.ravel(), alpha=0.25)[0])
    print("Mutual Information component-error: %f" % shuffle_test(mi, results_dict["c"],
                                                                       np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0])
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["c"], linewidth=1)
    #axes[1].axhline(y=max_test_lof, linewidth=2, linestyle="--", color="grey")
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Internal States")
    plt.show()

    print(evaluate_classification(y_true=labels, y_pred=(results_dict["c_"] < 0.5)))
    print("Mutual Information component-trigger: %f" % shuffle_test(mi, results_dict["c_"], Y_trigger.ravel(), alpha=0.25)[0])
    print("Mutual Information component-hypothesis: %f" % shuffle_test(mi, results_dict["c_"], hypothesis.ravel(), alpha=0.25)[0])
    print("Mutual Information component-error: %f" % shuffle_test(mi, results_dict["c_"],
                                                                       np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0])
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["c_"], linewidth=1)
    #axes[1].axhline(y=max_test_lof, linewidth=2, linestyle="--", color="grey")
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Candidate States")
    plt.show()

    print(evaluate_classification(y_true=labels, y_pred=(results_dict["i"] < 0.5)))
    print("Mutual Information component-trigger: %f" % shuffle_test(mi, results_dict["i"], Y_trigger.ravel(), alpha=0.25)[0])
    print("Mutual Information component-hypothesis: %f" % shuffle_test(mi, results_dict["i"], hypothesis.ravel(), alpha=0.25)[0])
    print("Mutual Information component-error: %f" % shuffle_test(mi, results_dict["i"],
                                                                       np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0])
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["i"], linewidth=1)
    #axes[1].axhline(y=max_test_lof, linewidth=2, linestyle="--", color="grey")
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Input Gates")
    plt.show()

    print(evaluate_classification(y_true=labels, y_pred=(results_dict["f"] < 0.5)))
    print("Mutual Information component-trigger: %f)" % shuffle_test(mi, results_dict["f"], Y_trigger.ravel(), alpha=0.25)[0])
    print("Mutual Information component-hypothesis: %f" % shuffle_test(mi, results_dict["f"], hypothesis.ravel(), alpha=0.25)[0])
    print("Mutual Information component-error: %f" % shuffle_test(mi, results_dict["f"],
                                                                       np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0])
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["f"], linewidth=1)
    #axes[1].axhline(y=max_test_lof, linewidth=2, linestyle="--", color="grey")
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Forget Gates")
    plt.show()

    print(evaluate_classification(y_true=labels, y_pred=(results_dict["o"] < 0.5)))
    print("Mutual Information component-trigger: %f" % shuffle_test(mi, results_dict["o"], Y_trigger.ravel(), alpha=0.25)[0])
    print("Mutual Information component-hypothesis: %f" % shuffle_test(mi, results_dict["o"], hypothesis.ravel(), alpha=0.25)[0])
    print("Mutual Information component-error: %f" % shuffle_test(mi, results_dict["o"],
                                                                       np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0])
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["o"], linewidth=1)
    #axes[1].axhline(y=max_test_lof, linewidth=2, linestyle="--", color="grey")
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Output Gates")
    plt.show()

    print(evaluate_classification(y_true=labels, y_pred=(results_dict["ifo"] < 0.5)))
    print("Mutual Information component-trigger: %f" % shuffle_test(mi, results_dict["ifo"], Y_trigger.ravel(), alpha=0.25)[0])
    print("Mutual Information component-hypothesis: %f" % shuffle_test(mi, results_dict["ifo"], hypothesis.ravel(), alpha=0.25)[0])
    print("Mutual Information component-error: %f" % shuffle_test(mi, results_dict["ifo"],
                                                                       np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0])
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["ifo"], linewidth=1)
    # axes[1].axhline(y=max_test_lof, linewidth=2, linestyle="--", color="grey")
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Avg. LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Input/Forget/Output Gates")
    plt.show()