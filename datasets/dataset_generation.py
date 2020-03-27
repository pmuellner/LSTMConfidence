import numpy as np
from utils.other import uniform
import matplotlib.pyplot as plt
import random
import pandas as pd
import os

class Outlier:
    def __init__(self, size):
        self.size = size

    def generate(self, data, constant=False, finetuning=None):
        data = data.copy()

        if finetuning == None:
            finetuning = np.ones(shape=(self.size,))
        else:
            finetuning = np.asarray(finetuning)

        outlier_start_idx = random.choice([i for i in range(0, len(data) - self.size + 1, self.size)])
        idxs = [i for i in range(outlier_start_idx, outlier_start_idx + self.size)]

        tuning_param = finetuning
        sign = random.choice([+1, -1])
        window = data[outlier_start_idx:outlier_start_idx + self.size]
        if sign == +1:
            to_be_added = max(data) - max(window)
            extremum = np.argmax(window)
        else:
            to_be_added = min(window) - min(data)
            extremum = np.argmin(window)

        if constant:
            generated_outlier = np.ones(self.size) * data[outlier_start_idx]
        else:
            add = np.array(uniform(low=0, high=to_be_added, n=self.size))
            generated_outlier = np.random.choice(window, self.size, replace=True) + add * sign * tuning_param
            generated_outlier[extremum] = window[extremum] + to_be_added * sign * tuning_param

        data_with_outliers = np.asarray(data)
        data_with_outliers[idxs] = generated_outlier

        return data_with_outliers, idxs

class SyntheticData:
    def __init__(self, kind, noise=0.0):
        self.kind = str(kind).lower()
        self.noise = noise

    def generate(self, n_periods_train, n_periods_test, prec):
        x_range_train = int(n_periods_train * 2 * np.pi)
        x_range_test = int(n_periods_test * 2 * np.pi)

        self.n_train = n_periods_train * prec
        self.n_test = n_periods_test * prec

        if self.kind == "sinus":
            return self._generate_sinus(x_range_train, x_range_test)
        else:
            raise NotImplementedError

    def _generate_sinus(self, range_train, range_test):
        x_train = np.linspace(0, range_train, self.n_train)
        x_train.sort()
        y_train = np.sin(x_train)

        x_test = np.linspace(0, range_test, self.n_test)
        x_test.sort()
        y_test = np.sin(x_test)

        x_trigger = np.linspace(0, range_test, self.n_test)
        x_trigger.sort()
        y_trigger = np.sin(x_trigger)

        self.y_train = y_train + np.random.randn(len(y_train)) * self.noise
        self.y_test = y_test + np.random.randn(len(y_test)) * self.noise
        self.y_trigger = y_trigger + np.random.randn(len(y_trigger)) * self.noise

        return self.y_train, self.y_test, self.y_trigger


if __name__ == "__main__":
    synth_data = SyntheticData(kind="sinus", noise=0.2)
    train, test, trigger = synth_data.generate(n_periods_train=25, n_periods_test=5, prec=100)

    #train = pd.read_csv("original_datasets/monthly_sunspots/train.csv", squeeze=True).values
    #test = pd.read_csv("original_datasets/monthly_sunspots/test.csv", squeeze=True).values
    #trigger = pd.read_csv("original_datasets/monthly_sunspots/outl.csv", squeeze=True).values

    for i in range(10):
        outlier = Outlier(size=20)
        outl_data, outl_idxs = outlier.generate(data=trigger, finetuning=[1], constant=False)
        labels = np.zeros(len(trigger))
        labels[outl_idxs] = 1

        path = "sinusoids/data" + str(i) + "/"
        os.mkdir(path)

        plt.plot(trigger, label="Original")
        plt.plot(outl_idxs, outl_data[outl_idxs], label="Outlier")
        plt.legend(loc="upper right")
        plt.savefig(path + "outliers.png")

        np.save(path + "train.npy", train)
        np.save(path + "test.npy", test)
        np.save(path + "outl.npy", trigger)
        np.save(path + "labels.npy", labels)