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
from scipy.special import erf, erfinv
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats.stats import pearsonr
from NPEET.npeet.entropy_estimators import mi, kldiv, shuffle_test, entropy, centropy
import os
from sklearn.preprocessing import MinMaxScaler
from glob import glob

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

def train_and_eval(X_train, Y_train, X_test, Y_test, X_trigger, Y_trigger, look_back, n_hidden):
    X = tf.placeholder(dtype=tf.float64, shape=[None, look_back, 1])
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

    weights_trigger = sess.run(W, feed_dict={X: X_trigger, Y: Y_trigger})
    weights_test = sess.run(W, feed_dict={X: X_test, Y: Y_test})

    biases_trigger = sess.run(b, feed_dict={X: X_trigger, Y: Y_trigger})
    biases_test = sess.run(b, feed_dict={X: X_test, Y: Y_test})

    i, f, o = get_gates(look_back=lb)
    c_ = get_candidate(look_back=lb)
    components_trigger = sess.run([h, c, c_, i, f, o], feed_dict={X: X_trigger, Y: Y_trigger})
    components_trigger = [component.T for component in components_trigger]
    components_test = sess.run([h, c, c_, i, f, o], feed_dict={X: X_test, Y: Y_test})
    components_test = [component.T for component in components_test]

    hypothesis = sess.run(preds, feed_dict={X: X_trigger, Y: Y_trigger})

    tf.reset_default_graph()
    sess.close()

    return components_trigger, weights_trigger, biases_trigger, components_test, weights_test, biases_test, hypothesis

def load_datasets(dataset_name):
    DATASETS_PREFIX = "datasets/"
    datasets = []
    for path in glob(DATASETS_PREFIX + dataset_name + "/*/"):
        data_dict = dict()
        data_dict["train"] = MinMaxScaler().fit_transform(np.load(path + "train.npy").reshape(-1, 1)).ravel()
        data_dict["test"] = MinMaxScaler().fit_transform(np.load(path + "test.npy").reshape(-1, 1)).ravel()
        data_dict["trigger"] = MinMaxScaler().fit_transform(np.load(path + "outl.npy").reshape(-1, 1)).ravel()
        data_dict["labels"] = np.load(path + "labels.npy")
        datasets.append(data_dict)

    if len(datasets) == 0:
        print("No Datasets found!")

    return datasets

def get_mi_of_components(X_train, Y_train, X_test, Y_test, X_trigger, Y_trigger, look_back, n_hidden):
    X = tf.placeholder(dtype=tf.float64, shape=[None, look_back, 1])
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
    for epoch in range(n_iter):
        sess.run(train_step, feed_dict={X: X_train, Y: Y_train})

    i, f, o = get_gates(look_back=lb)
    c_ = get_candidate(look_back=lb)
    components_test = sess.run([h, c, c_, i, f, o], feed_dict={X: X_test, Y: Y_test})
    components_test = [component.T for component in components_test]

    weights_test = sess.run(W, feed_dict={X: X_test, Y: Y_test})

    hypothesis = sess.run(preds, feed_dict={X: X_test, Y: Y_test})
    names = ["h", "c", "c_", "i", "f", "o"]
    for name, component in zip(names, components_test):
        x = component.T.dot(weights_test)
        trigger_avg_mi = shuffle_test(mi, x, Y_test.ravel(), alpha=0.25)[0]
        hyp_avg_mi = shuffle_test(mi, x, hypothesis.ravel(), alpha=0.25)[0]
        err_avg_mi = shuffle_test(mi, x, np.abs(hypothesis - Y_test).ravel(), alpha=0.25)[0]

        print("I(trigger; " + name + "): " + str(trigger_avg_mi))
        print("I(hypothesis; " + name + "): " + str(hyp_avg_mi))
        print("I(error; " + name + "): " + str(err_avg_mi))

    tf.reset_default_graph()
    sess.close()



def run(X_train, Y_train, X_test, Y_test, X_trigger, Y_trigger, labels, look_back, n_hidden, verbose=True):
    list_of_confidence_per_run = []
    for i in range(5):
        components_trigger, weights_trigger, biases_trigger, components_test, weights_test, biases_test, hypothesis \
            = train_and_eval(X_train, Y_train, X_test, Y_test, X_trigger, Y_trigger, look_back, n_hidden)

        confidence_dict = {}
        names = ["h", "c", "c_", "i", "f", "o", "ifo"]
        components_trigger.append(np.dstack((components_trigger[-1], components_trigger[-2], components_trigger[-3])))
        components_test.append(np.dstack((components_test[-1], components_test[-2], components_test[-3])))


        for name, component_trigger, component_test in zip(names, components_trigger, components_test):
            lof_trigger_mat = np.zeros((component_trigger.shape[0], component_trigger.shape[1]))
            lof_test_mat = np.zeros((component_test.shape[0], component_test.shape[1]))

            for neuron in range(n_hidden):
                values_trigger = component_trigger[neuron]
                values_test = component_test[neuron]
                #lof = LocalOutlierProbability(values).fit().local_outlier_probabilities

                if name in ["f", "i", "o", "ifo"]:
                    activation = sigmoid
                elif name in ["c_"]:
                    activation = np.tanh
                else:
                    activation = lambda x: x

                if name != "ifo":
                    lof_trigger = LocalOutlierFactor(n_neighbors=15, algorithm="brute").fit(
                        activation(values_trigger).reshape(-1, 1)).negative_outlier_factor_ * -1
                    lof_test = LocalOutlierFactor(n_neighbors=15, algorithm="brute").fit(
                        activation(values_test).reshape(-1, 1)).negative_outlier_factor_ * -1
                    """lof_trigger = LocalOutlierProbability(values_trigger.reshape(-1, 1), n_neighbors=15).fit().local_outlier_probabilities
                    lof_test = LocalOutlierProbability(values_test.reshape(-1, 1), n_neighbors=15).fit().local_outlier_probabilities
                    """
                else:
                    lof_trigger = LocalOutlierFactor(n_neighbors=15, algorithm="brute").fit(
                        activation(values_trigger)).negative_outlier_factor_ * -1
                    lof_test = LocalOutlierFactor(n_neighbors=15, algorithm="brute").fit(
                        activation(values_test)).negative_outlier_factor_ * -1
                    """lof_trigger = LocalOutlierProbability(values_trigger,
                                                          n_neighbors=15).fit().local_outlier_probabilities
                    lof_test = LocalOutlierProbability(values_test,
                                                       n_neighbors=15, ).fit().local_outlier_probabilities
                    """

                lof_trigger_mat[neuron] = lof_trigger
                lof_test_mat[neuron] = lof_test

            w_trigger = np.abs(weights_trigger) / np.sum(np.abs(weights_trigger))
            w_test = np.abs(weights_test) / np.sum(np.abs(weights_test))
            #overall_lof_trigger = lof_trigger_mat.T.dot(w_trigger).ravel()
            #overall_lof_test = lof_test_mat.T.dot(w_test).ravel()

            overall_lof_trigger = lof_trigger_mat.T.dot(w_trigger) + biases_trigger
            overall_lof_test = lof_test_mat.T.dot(w_test) + biases_test

            conservativeness = 0.25
            scaler = MinMaxScaler(feature_range=(0, erfinv(np.array(conservativeness))))
            x_test = scaler.fit_transform(overall_lof_test.reshape(-1, 1))
            x_trigger = scaler.transform(overall_lof_trigger.reshape(-1, 1))

            error_probability = erf((x_trigger - np.mean(x_trigger)))
            confidence = 1 - error_probability
            confidence_dict[name] = confidence
        list_of_confidence_per_run.append(confidence_dict)

    results_dict = mean_of_dictionaries(*list_of_confidence_per_run)

    """fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(np.abs(Y_trigger - hypothesis),
                 linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Mean Absolute Error")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("X")
    plt.show()

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(LocalOutlierFactor().fit(Y_trigger).negative_outlier_factor_ * -1,
                 linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("LOF")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("X")
    plt.show()


    results_dict["y"] = (LocalOutlierFactor().fit(Y_trigger).negative_outlier_factor_ * -1).reshape(-1, 1)
    y_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["y"] > 1.5))
    #y_eval["I(component; trigger)"] = shuffle_test(mi, results_dict["y"], Y_trigger.ravel(), alpha=0.25)[0]
    #y_eval["I(component; hypothesis)"] = shuffle_test(mi, results_dict["y"], hypothesis.ravel(), alpha=0.25)[0]
    #y_eval["I(component; error)"] = shuffle_test(mi, results_dict["y"], np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0]

    h_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["h"] < 0.5))
    #print("Mutual Information component-trigger: %f" % shuffle_test(mi, results_dict["h"], Y_trigger.ravel(), alpha=0.25)[0])
    #print("Mutual Information component-hypothesis: %f" % shuffle_test(mi, results_dict["h"], hypothesis.ravel(), alpha=0.25)[0])
    #print("Mutual Information component-error: %f" % shuffle_test(mi, results_dict["h"], np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0])
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["h"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Confidence")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Hidden States")
    plt.show()

    c_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["c"] < 0.5))
    #print("Mutual Information component-trigger: %f" % shuffle_test(mi, results_dict["c"], Y_trigger.ravel(), alpha=0.25)[0])
    #print("Mutual Information component-hypothesis: %f" % shuffle_test(mi, results_dict["c"], hypothesis.ravel(), alpha=0.25)[0])
    #print("Mutual Information component-error: %f" % shuffle_test(mi, results_dict["c"], np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0])
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["c"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Confidence")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Internal States")
    plt.show()

    cand_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["c_"] < 0.5))
    #print("Mutual Information component-trigger: %f" % shuffle_test(mi, results_dict["c_"], Y_trigger.ravel(), alpha=0.25)[0])
    #print("Mutual Information component-hypothesis: %f" % shuffle_test(mi, results_dict["c_"], hypothesis.ravel(), alpha=0.25)[0])
    #print("Mutual Information component-error: %f" % shuffle_test(mi, results_dict["c_"], np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0])
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["c_"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Confidence")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Candidate States")
    plt.show()

    i_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["i"] < 0.5))
    #print("Mutual Information component-trigger: %f" % shuffle_test(mi, results_dict["i"], Y_trigger.ravel(), alpha=0.25)[0])
    #print("Mutual Information component-hypothesis: %f" % shuffle_test(mi, results_dict["i"], hypothesis.ravel(), alpha=0.25)[0])
    #print("Mutual Information component-error: %f" % shuffle_test(mi, results_dict["i"],  np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0])
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["i"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Confidence")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Input Gates")
    plt.show()

    f_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["f"] < 0.5))
    #print("Mutual Information component-trigger: %f)" % shuffle_test(mi, results_dict["f"], Y_trigger.ravel(), alpha=0.25)[0])
    #print("Mutual Information component-hypothesis: %f" % shuffle_test(mi, results_dict["f"], hypothesis.ravel(), alpha=0.25)[0])
    #print("Mutual Information component-error: %f" % shuffle_test(mi, results_dict["f"], np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0])
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["f"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Confidence")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Forget Gates")
    plt.show()

    o_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["o"] < 0.5))
    #print("Mutual Information component-trigger: %f" % shuffle_test(mi, results_dict["o"], Y_trigger.ravel(), alpha=0.25)[0])
    #print("Mutual Information component-hypothesis: %f" % shuffle_test(mi, results_dict["o"], hypothesis.ravel(), alpha=0.25)[0])
    #print("Mutual Information component-error: %f" % shuffle_test(mi, results_dict["o"],np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0])
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["o"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Confidence")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Output Gates")
    plt.show()

    ifo_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["ifo"] < 0.5))
    #print("Mutual Information component-trigger: %f" % shuffle_test(mi, results_dict["ifo"], Y_trigger.ravel(), alpha=0.25)[0])
    #print("Mutual Information component-hypothesis: %f" % shuffle_test(mi, results_dict["ifo"], hypothesis.ravel(), alpha=0.25)[0])
    #print("Mutual Information component-error: %f" % shuffle_test(mi, results_dict["ifo"], np.abs(hypothesis - Y_trigger).ravel(), alpha=0.25)[0])
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["ifo"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Confidence")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Input/Forget/Output Gates")
    plt.show()"""

    fig, axes = plt.subplots(2, 1, sharex=True)
    cbar_ax = fig.add_axes([.91, .10, .03, .4])
    axes[0].plot(Y_trigger, linewidth=1)
    heatmap = np.zeros((7, len(Y_trigger)))
    heatmap[0] = np.array(results_dict["h"]).ravel()
    heatmap[1] = np.array(results_dict["c"]).ravel()
    heatmap[2] = np.array(results_dict["c_"]).ravel()
    heatmap[3] = np.array(results_dict["i"]).ravel()
    heatmap[4] = np.array(results_dict["f"]).ravel()
    heatmap[5] = np.array(results_dict["o"]).ravel()
    heatmap[6] = np.array(results_dict["ifo"]).ravel()
    sns.heatmap(heatmap, ax=axes[1], cmap="Reds_r", vmin=0, vmax=1, cbar=True, cbar_ax=cbar_ax)
    axes[1].set_yticklabels([r"$h$", r"$c$", r"$\bar{c}$", r"$i$", r"$f$", r"$o$", r"$[ifo]$"], rotation=0)
    plt.show()

    plt.figure(figsize=(6, 3))
    sns.heatmap(np.array(results_dict["ifo"]).reshape(1, -1), cbar=True, vmin=0, vmax=1, cmap="Reds_r", alpha=1, cbar_kws={'label': 'confidence'})
    plt.ylim([0, 1])
    plt.yticks([])
    plt.xticklabels(rotation=0)
    plt.plot(Y_trigger, linewidth=1)
    plt.xticklabels(rotation=0)
    plt.xlabel(r"time $t$")
    plt.title(r"$[ifo]$")
    plt.tight_layout()
    plt.show()

    return y_eval, h_eval, c_eval, cand_eval, i_eval, f_eval, o_eval, ifo_eval


if __name__ == "__main__":
    lb = 5
    n_hidden = 10
    datasets = load_datasets("sunspot")
    dataset = datasets[8]

    X_train, Y_train = create_sequences(dataset["train"], look_back=lb)
    X_test, Y_test = create_sequences(dataset["test"], look_back=lb)
    X_trigger, Y_trigger = create_sequences(dataset["trigger"], look_back=lb)
    labels = dataset["labels"][lb+1:]


    y_eval, h_eval, c_eval, cand_eval, i_eval, f_eval, o_eval, ifo_eval = run(X_train, Y_train,
                                                                          X_test, Y_test,
                                                                          X_trigger, Y_trigger,
                                                                          labels=labels,
                                                                          verbose=False, look_back=lb, n_hidden=n_hidden)

    print("Output y")
    print(y_eval)
    print("Hidden State h")
    print(h_eval)
    print("Internal State c")
    print(c_eval)
    print("Candidate State c_")
    print(cand_eval)
    print("Input Gate i")
    print(i_eval)
    print("Forget Gate f")
    print(f_eval)
    print("Output Gate o")
    print(o_eval)
    print("Input/Forget/Output Gates ifo")
    print(ifo_eval)