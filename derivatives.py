import numpy as np
from activation_functions import *
from loss_functions import *
"""
  DERIVATIVES OF ACTIVATION AND LOSS FUNCTIONS
"""
def identity_derivative(x):
    return np.ones_like(x)

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def mean_squared_error_derivative(y_true, y_pred):
    return y_pred - y_true

def cross_entropy_loss_derivative(y_true, y_pred):
    return -y_true / (y_pred + 1e-9)

def softmax_derivative(inp:np.array):
    derivates = []
    if(len(inp.shape) == 1):
      S_vector = inp.reshape(-1, 1)
      derivates = np.diag(inp) - np.dot(S_vector, S_vector.T)
    elif(len(inp.shape) == 2):
      for i in range(inp.shape[0]):
        S_vector = inp[i].reshape(-1, 1)
        derivates.append(np.diag(inp[i]) - np.dot(S_vector, S_vector.T))

    return np.array(derivates)
