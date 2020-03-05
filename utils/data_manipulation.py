import numpy as np

def create_sequences(data, look_back):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        x = data[i:i+look_back]
        y = data[i+look_back]
        X.append(x)
        Y.append(y)
    X = np.array(X)[..., np.newaxis]
    Y = np.array(Y)[..., np.newaxis]

    return X, Y