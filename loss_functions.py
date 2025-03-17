import numpy as np
"""
  LOSS FUNCTIONS
"""
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9), axis=-1)    # To avoid log 0, 1e-9 added to y_pred
