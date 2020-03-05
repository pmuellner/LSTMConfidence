import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

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

        """curvature = np.gradient(np.gradient(y_trigger))
        turning_points = np.argwhere(np.diff(np.sign(curvature)) != 0).ravel().tolist()
        trigger_start_idx = random.choice(range(len(turning_points[:-1])))
        trigger_start, trigger_end = turning_points[trigger_start_idx], turning_points[trigger_start_idx+1]

        y_trigger[trigger_start:trigger_end + 1] = np.random.randn(trigger_end+1 - trigger_start) * 0.5
        if window_size > 0:
            y_trigger = np.convolve(y_trigger, np.ones(window_size) / window_size, mode="same")"""



        self.y_train = y_train + np.random.randn(len(y_train)) * self.noise
        self.y_test = y_test + np.random.randn(len(y_test)) * self.noise
        self.y_trigger = y_trigger + np.random.randn(len(y_trigger)) * self.noise

        self._add_outlier(size=10, scaling=.8)

        return self.y_train, self.y_test, self.y_trigger

    def _add_outlier(self, size, scaling):
        outlier = np.random.uniform(low=self.y_trigger.min(), high=self.y_trigger.max(), size=size) * scaling
        outlier_start_idx = np.random.randint(low=0, high=self.n_test-size)
        outlier_end_idx = outlier_start_idx + size
        self.y_trigger[outlier_start_idx:outlier_end_idx] = outlier


if __name__ == "__main__":
    synth_data = SyntheticData(kind="sinus", noise=0.4)
    train, test, trigger = synth_data.generate(n_periods_train=25, n_periods_test=5, prec=100)
    print(len(train), len(test), len(trigger))
    plt.plot(test, label="Test")
    plt.plot(trigger, label="Trigger")
    plt.legend()
    plt.show()

    data_dict = {"train": train, "test": test, "trigger": trigger}
    with open("sinusoids/data1.pkl", "wb") as f:
        pickle.dump(data_dict, f)


