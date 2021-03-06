import tensorflow as tf
from utils.data_manipulation import create_sequences
from custom_cells.lstm import LSTM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from collections import defaultdict
from scipy.special import erf, erfinv
from sklearn.metrics import precision_score, recall_score, f1_score
from NPEET.npeet.entropy_estimators import mi, shuffle_test
from sklearn.preprocessing import MinMaxScaler
from glob import glob
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl

mpl.rcParams.update({'font.size': 12})

plt.style.use("seaborn-colorblind")
sns.set_style("ticks")


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

    return {"Precision": precision, "Recall": recall, "F1-score": f1, "Average Precision": ap}


def mean_of_dictionaries(*args):
    summation = defaultdict(list)
    for dictionary in args:
        for key in dictionary:
            summation[key].append(dictionary[key])

    for key in summation:
        summation[key] = np.array(summation[key]).mean(axis=0)

    return dict(summation)


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

def confidence(lof_trigger, lof_reference, conservativeness):
    scaler = MinMaxScaler(feature_range=(0, erfinv(np.array(conservativeness))))
    x_test = scaler.fit_transform(lof_reference.reshape(-1, 1))
    x_trigger = scaler.transform(lof_trigger.reshape(-1, 1))

    error_probability = erf((x_trigger - np.mean(x_trigger)))
    confidence = 1 - error_probability

    return confidence

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

    hypothesis_trigger = sess.run(preds, feed_dict={X: X_trigger, Y: Y_trigger})
    hypothesis_test = sess.run(preds, feed_dict={X: X_test, Y: Y_test})

    tf.reset_default_graph()
    sess.close()

    return components_trigger, weights_trigger, biases_trigger, components_test, weights_test, biases_test, hypothesis_trigger, hypothesis_test

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


def run(X_train, Y_train, X_test, Y_test, X_trigger, Y_trigger, labels, look_back, n_hidden, beta):
    list_of_confidence_per_run = []
    confidence_z_per_run = []
    beta = 0.33
    for i in range(5):
        components_trigger, weights_trigger, biases_trigger, components_test, weights_test, biases_test, z_trigger, z_test = train_and_eval(X_train, Y_train, X_test, Y_test, X_trigger, Y_trigger, look_back, n_hidden)

        lof_z_trigger = LocalOutlierFactor(n_neighbors=15, algorithm="brute").fit(
            z_trigger.reshape(-1, 1)).negative_outlier_factor_ * -1
        lof_z_test = LocalOutlierFactor(n_neighbors=15, algorithm="brute").fit(
            z_test.reshape(-1, 1)).negative_outlier_factor_ * -1

        confidence_z = confidence(lof_trigger=lof_z_trigger, lof_reference=lof_z_test, conservativeness=beta)
        confidence_z_per_run.append(confidence_z)

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

                if name != "ifo":
                    lof_trigger = LocalOutlierFactor(n_neighbors=15, algorithm="brute").fit(
                        values_trigger.reshape(-1, 1)).negative_outlier_factor_ * -1
                    lof_test = LocalOutlierFactor(n_neighbors=15, algorithm="brute").fit(
                        values_test.reshape(-1, 1)).negative_outlier_factor_ * -1
                    """lof_trigger = LocalOutlierProbability(values_trigger.reshape(-1, 1), n_neighbors=15).fit().local_outlier_probabilities
                    lof_test = LocalOutlierProbability(values_test.reshape(-1, 1), n_neighbors=15).fit().local_outlier_probabilities
                    """
                else:
                    lof_trigger = LocalOutlierFactor(n_neighbors=15, algorithm="brute").fit(
                        values_trigger).negative_outlier_factor_ * -1
                    lof_test = LocalOutlierFactor(n_neighbors=15, algorithm="brute").fit(
                        values_test).negative_outlier_factor_ * -1
                    """lof_trigger = LocalOutlierProbability(values_trigger,
                                                          n_neighbors=15).fit().local_outlier_probabilities
                    lof_test = LocalOutlierProbability(values_test,
                                                       n_neighbors=15, ).fit().local_outlier_probabilities
                    """

                lof_trigger_mat[neuron] = lof_trigger
                lof_test_mat[neuron] = lof_test

            w_trigger = np.abs(weights_trigger) / np.sum(np.abs(weights_trigger))
            w_test = np.abs(weights_test) / np.sum(np.abs(weights_test))

            overall_lof_trigger = lof_trigger_mat.T.dot(w_trigger) + biases_trigger
            overall_lof_test = lof_test_mat.T.dot(w_test) + biases_test

            confidence_dict[name] = confidence(lof_trigger=overall_lof_trigger, lof_reference=overall_lof_test, conservativeness=beta)
        list_of_confidence_per_run.append(confidence_dict)

    results_dict = mean_of_dictionaries(*list_of_confidence_per_run)
    results_dict["z"] = np.mean(confidence_z_per_run, axis=0)


    lof_x_trigger = (LocalOutlierFactor().fit(Y_trigger).negative_outlier_factor_ * -1).reshape(-1, 1)
    lof_x_test = (LocalOutlierFactor().fit(Y_test).negative_outlier_factor_ * -1).reshape(-1, 1)
    results_dict["x"] = confidence(lof_trigger=lof_x_trigger, lof_reference=lof_x_test, conservativeness=0.25)
    x_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["x"] < 0.5))
    z_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["z"] < 0.5))

    h_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["h"] < 0.5))
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["h"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Confidence")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Hidden States")
    plt.show()

    c_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["c"] < 0.5))
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["c"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Confidence")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Internal States")
    plt.show()

    cand_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["c_"] < 0.5))
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["c_"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Confidence")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Candidate States")
    plt.show()

    i_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["i"] < 0.5))
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["i"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Confidence")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Input Gates")
    plt.show()

    f_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["f"] < 0.5))
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["f"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Confidence")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Forget Gates")
    plt.show()

    o_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["o"] < 0.5))
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["o"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Confidence")
    plt.xlabel("Time " + r"$t$")
    plt.suptitle("Output Gates")
    plt.show()

    ifo_eval = evaluate_classification(y_true=labels, y_pred=(results_dict["ifo"] < 0.5))
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(Y_trigger, linewidth=1)
    axes[1].plot(results_dict["ifo"], linewidth=1)
    axes[0].set_ylabel("Time Series")
    axes[1].set_ylabel("Confidence")
    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(50))
    axes[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xlabel(r"Time $t$")
    axes[0].set_title("Input/Forget/Output Gates")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, 1, sharex=True)
    divider = make_axes_locatable(axes[1])
    cbar_ax = divider.append_axes("right", size="5%", pad=0.05)

    divider = make_axes_locatable(axes[0])
    cbar_ax2_helper = divider.append_axes("right", size="5%", pad=0.05)
    cbar_ax2_helper.set_visible(False)
    axes[0].plot(Y_trigger, linewidth=1)
    heatmap = np.zeros((9, len(Y_trigger)))
    heatmap[0] = np.array(results_dict["x"]).ravel()
    heatmap[1] = np.array(results_dict["z"]).ravel()
    heatmap[2] = np.array(results_dict["h"]).ravel()
    heatmap[3] = np.array(results_dict["c"]).ravel()
    heatmap[4] = np.array(results_dict["c_"]).ravel()
    heatmap[5] = np.array(results_dict["i"]).ravel()
    heatmap[6] = np.array(results_dict["f"]).ravel()
    heatmap[7] = np.array(results_dict["o"]).ravel()
    heatmap[8] = np.array(results_dict["ifo"]).ravel()
    g = sns.heatmap(heatmap, ax=axes[1], cmap="Reds_r", vmin=0, vmax=1, cbar=True, cbar_ax=cbar_ax, cbar_kws={'label': 'Confidence'})
    axes[1].set_yticklabels([r"$X$", r"$Z$", r"$h_t$", r"$c_t$", r"$\bar{c}_t$", r"$i_t$", r"$f_t$", r"$o_t$", r"$ifo_t$"], rotation=0)
    axes[1].set_xticklabels(range(heatmap.shape[1]), rotation=0)
    g.xaxis.set_major_locator(ticker.MultipleLocator(50))
    g.xaxis.set_major_formatter(ticker.ScalarFormatter())
    axes[1].axhline(y=1, c="white", linewidth=2)
    axes[1].axhline(y=2, c="white", linewidth=2)
    axes[1].axhline(y=3, c="white", linewidth=2)
    axes[1].axhline(y=4, c="white", linewidth=2)
    axes[1].axhline(y=5, c="white", linewidth=2)
    axes[1].axhline(y=6, c="white", linewidth=2)
    axes[1].axhline(y=7, c="white", linewidth=2)
    axes[1].axhline(y=8, c="white", linewidth=2)
    axes[0].set_ylabel("Time Series")
    axes[1].set_xlabel(r"Time $t$")
    plt.show()

    plt.figure(figsize=(6, 3))
    g = sns.heatmap(np.array(results_dict["ifo"]).reshape(1, -1), cbar=True, vmin=0, vmax=1, cmap="Reds_r", alpha=1, cbar_kws={'label': 'Confidence'})
    g.xaxis.set_major_locator(ticker.MultipleLocator(50))
    g.xaxis.set_major_formatter(ticker.ScalarFormatter())
    g.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    g.yaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xticks(rotation=0)
    plt.ylabel("Time Series")
    plt.yticks(rotation=0)
    plt.ylim([0, 1])
    plt.plot(Y_trigger, linewidth=.5)
    plt.xlabel(r"Time $t$")
    plt.title("Input/Forget/Output Gate")
    plt.tight_layout()
    plt.show()

    return x_eval, z_eval, h_eval, c_eval, cand_eval, i_eval, f_eval, o_eval, ifo_eval


if __name__ == "__main__":
    lb = 5
    n_hidden = 10
    datasets = load_datasets("ecg0")
    dataset = datasets[2]

    X_train, Y_train = create_sequences(dataset["train"], look_back=lb)
    X_test, Y_test = create_sequences(dataset["test"], look_back=lb)
    X_trigger, Y_trigger = create_sequences(dataset["trigger"], look_back=lb)
    labels = dataset["labels"][lb+1:]


    x_eval, z_eval, h_eval, c_eval, cand_eval, i_eval, f_eval, o_eval, ifo_eval = run(X_train, Y_train,
                                                                          X_test, Y_test,
                                                                          X_trigger, Y_trigger,
                                                                          labels=labels,
                                                                          look_back=lb,
                                                                          n_hidden=n_hidden,
                                                                          beta=0.33)

    print("Input x")
    print(x_eval)
    print("Output z")
    print(z_eval)
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