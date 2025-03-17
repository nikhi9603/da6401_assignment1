import numpy as np
"""
  ACTIVATION FUNCTIONS
"""
def identity(x):
    return x

def sigmoid(x):
    # x = np.clip(x,-10,10)
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    # print(x)
    # x = np.clip(x, -200,200)
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical Stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
