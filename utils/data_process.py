import numpy as np

def load_data():
    f = np.load('../data/train.npz')
    x_train = f['a']
    y_train = f['b']

    f = np.load('../data/validate.npz')
    x_validate = f['a']
    y_validate = f['b']

    f = np.load('../data/ood_sample.npz')
    x_ood = f['a']
    return (x_train, y_train, x_validate, y_validate, x_ood)
