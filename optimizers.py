import numpy as np
"""
  OPTIMIZERS UPDATE RULES
"""

# STOCHASTIC GRADIENT DESCENT
def sgd(optimizer_input_dict, wts_bias_history_dict, itr=None):
  # cant update weights in one single matrix op as dimensions of weights can be different in each layer
  for i in range(optimizer_input_dict["n_hiddenLayers"]+1):
    # weight decay term added additionally to the formula in slides
    wts_bias_history_dict["weights"][i] = wts_bias_history_dict["weights"][i] - optimizer_input_dict["learning_rate"] * (wts_bias_history_dict["dw"][i])
    wts_bias_history_dict["biases"][i] = wts_bias_history_dict["biases"][i] - optimizer_input_dict["learning_rate"] * (wts_bias_history_dict["db"][i])

# MOMENTUM BASED GRADIENT DESCENT
def momentumGradientDescent(optimizer_input_dict, wts_bias_history_dict, itr=None):
  for i in range(optimizer_input_dict["n_hiddenLayers"]+1):
    wts_bias_history_dict["history_weights"][i] = (optimizer_input_dict["momentum"] * wts_bias_history_dict["history_weights"][i]) + wts_bias_history_dict["dw"][i]
    wts_bias_history_dict["weights"][i] = wts_bias_history_dict["weights"][i] - optimizer_input_dict["learning_rate"] * (wts_bias_history_dict["history_weights"][i])

    wts_bias_history_dict["history_biases"][i] = (optimizer_input_dict["momentum"] * wts_bias_history_dict["history_biases"][i]) + wts_bias_history_dict["db"][i]
    wts_bias_history_dict["biases"][i] = wts_bias_history_dict["biases"][i] - (optimizer_input_dict["learning_rate"] * wts_bias_history_dict["history_biases"][i])

# NAG(NESTEROV ACCELERATED GRADIENT DESCENT)
def nag(optimizer_input_dict, wts_bias_history_dict, itr=None):
  for i in range(optimizer_input_dict["n_hiddenLayers"]+1):
    # dw,db will contain lookahead gradients only since forward and backward propagations are implemented accordingly
    wts_bias_history_dict["history_weights"][i] = (optimizer_input_dict["momentum"] * wts_bias_history_dict["history_weights"][i]) + wts_bias_history_dict["dw"][i]
    wts_bias_history_dict["weights"][i] = wts_bias_history_dict["weights"][i] - optimizer_input_dict["learning_rate"] * (wts_bias_history_dict["history_weights"][i])

    wts_bias_history_dict["history_biases"][i] = (optimizer_input_dict["momentum"] * wts_bias_history_dict["history_biases"][i]) + wts_bias_history_dict["db"][i]
    wts_bias_history_dict["biases"][i] = wts_bias_history_dict["biases"][i] - optimizer_input_dict["learning_rate"] * wts_bias_history_dict["history_biases"][i]

# RMSPROP
def rmsProp(optimizer_input_dict, wts_bias_history_dict, itr=None):
  for i in range(optimizer_input_dict["n_hiddenLayers"]+1):
    wts_bias_history_dict["history_weights"][i] = (optimizer_input_dict["beta"] * wts_bias_history_dict["history_weights"][i]) + (1 - optimizer_input_dict["beta"]) * (wts_bias_history_dict["dw"][i] ** 2)
    # wts_bias_history_dict["weights"][i] = wts_bias_history_dict["weights"][i] - optimizer_input_dict["learning_rate"] *((wts_bias_history_dict["dw"][i]/np.sqrt(wts_bias_history_dict["history_weights"][i] + optimizer_input_dict["epsilon"])))  - (optimizer_input_dict["weight_decay"] * wts_bias_history_dict["weights"][i])
    wts_bias_history_dict["weights"][i] = wts_bias_history_dict["weights"][i] - ((optimizer_input_dict["learning_rate"]/np.sqrt(wts_bias_history_dict["history_weights"][i] + optimizer_input_dict["epsilon"])) * (wts_bias_history_dict["dw"][i]))


    wts_bias_history_dict["history_biases"][i] = (optimizer_input_dict["beta"] * wts_bias_history_dict["history_biases"][i]) + (1 - optimizer_input_dict["beta"]) * (wts_bias_history_dict["db"][i] ** 2)
    wts_bias_history_dict["biases"][i] = wts_bias_history_dict["biases"][i] - ((optimizer_input_dict["learning_rate"]/np.sqrt(wts_bias_history_dict["history_biases"][i] + optimizer_input_dict["epsilon"])) * (wts_bias_history_dict["db"][i]))

# ADAM
def adam(optimizer_input_dict, wts_bias_history_dict, itr=None):
  for i in range(optimizer_input_dict["n_hiddenLayers"]+1):
    wts_bias_history_dict["history_weights"][i] = (optimizer_input_dict["beta1"] * wts_bias_history_dict["history_weights"][i]) + (1 - optimizer_input_dict["beta1"]) * (wts_bias_history_dict["dw"][i])
    wts_bias_history_dict["second_history_weights"][i] = (optimizer_input_dict["beta2"] * wts_bias_history_dict["second_history_weights"][i]) + (1 - optimizer_input_dict["beta2"]) * (wts_bias_history_dict["dw"][i] ** 2)

    history_weights_hat = wts_bias_history_dict["history_weights"][i] / (1 - (optimizer_input_dict["beta1"] ** (itr)))
    second_history_weights_hat = wts_bias_history_dict["second_history_weights"][i] / (1 - (optimizer_input_dict["beta2"] ** (itr)))

    # wts_bias_history_dict["weights"][i] = wts_bias_history_dict["weights"][i] - (optimizer_input_dict["learning_rate"]*((history_weights_hat/(np.sqrt(second_history_weights_hat) + optimizer_input_dict["epsilon"])))) - ((optimizer_input_dict["weight_decay"] * wts_bias_history_dict["weights"][i]))
    wts_bias_history_dict["weights"][i] = wts_bias_history_dict["weights"][i] - ((optimizer_input_dict["learning_rate"]/(np.sqrt(second_history_weights_hat) + optimizer_input_dict["epsilon"])) * (history_weights_hat))

    wts_bias_history_dict["history_biases"][i] = (optimizer_input_dict["beta1"] * wts_bias_history_dict["history_biases"][i]) + (1 - optimizer_input_dict["beta1"]) * wts_bias_history_dict["db"][i]
    wts_bias_history_dict["second_history_biases"][i] = (optimizer_input_dict["beta2"] * wts_bias_history_dict["second_history_biases"][i]) + (1 - optimizer_input_dict["beta2"]) * (wts_bias_history_dict["db"][i] ** 2)

    history_biases_hat = wts_bias_history_dict["history_biases"][i] / (1 - (optimizer_input_dict["beta1"] ** (itr)))
    second_history_biases_hat = wts_bias_history_dict["second_history_biases"][i] / (1 - (optimizer_input_dict["beta2"] ** (itr)))

    wts_bias_history_dict["biases"][i] = wts_bias_history_dict["biases"][i] - ((optimizer_input_dict["learning_rate"]/(np.sqrt(second_history_biases_hat) + optimizer_input_dict["epsilon"])) * (history_biases_hat))

# NADAM
def nadam(optimizer_input_dict, wts_bias_history_dict, itr=None):
  for i in range(optimizer_input_dict["n_hiddenLayers"]+1):
    wts_bias_history_dict["history_weights"][i] = (optimizer_input_dict["beta1"] * wts_bias_history_dict["history_weights"][i]) + (1 - optimizer_input_dict["beta1"]) * (wts_bias_history_dict["dw"][i])
    wts_bias_history_dict["second_history_weights"][i] = (optimizer_input_dict["beta2"] * wts_bias_history_dict["second_history_weights"][i]) + (1 - optimizer_input_dict["beta2"]) * (wts_bias_history_dict["dw"][i] ** 2)

    history_weights_hat = wts_bias_history_dict["history_weights"][i] / (1 - (optimizer_input_dict["beta1"] ** itr))
    second_history_weights_hat = wts_bias_history_dict["second_history_weights"][i] / (1 - (optimizer_input_dict["beta2"] ** itr))

    lookahead_dw = optimizer_input_dict["beta1"] * history_weights_hat + (((1-optimizer_input_dict["beta1"])/(1-(optimizer_input_dict["beta1"] ** itr))) * wts_bias_history_dict["dw"][i])
    # wts_bias_history_dict["weights"][i] = wts_bias_history_dict["weights"][i] - (optimizer_input_dict["learning_rate"]*(lookahead_dw/(np.sqrt(second_history_weights_hat) + optimizer_input_dict["epsilon"]))) - ((optimizer_input_dict["weight_decay"] * wts_bias_history_dict["weights"][i]))
    wts_bias_history_dict["weights"][i] = wts_bias_history_dict["weights"][i] - ((optimizer_input_dict["learning_rate"]/(np.sqrt(second_history_weights_hat) + optimizer_input_dict["epsilon"])) * (lookahead_dw))


    wts_bias_history_dict["history_biases"][i] = (optimizer_input_dict["beta1"] * wts_bias_history_dict["history_biases"][i]) + (1 - optimizer_input_dict["beta1"]) * wts_bias_history_dict["db"][i]
    wts_bias_history_dict["second_history_biases"][i] = (optimizer_input_dict["beta2"] * wts_bias_history_dict["second_history_biases"][i]) + (1 - optimizer_input_dict["beta2"]) * (wts_bias_history_dict["db"][i] ** 2)

    history_biases_hat = wts_bias_history_dict["history_biases"][i] / (1 - (optimizer_input_dict["beta1"] ** itr))
    second_history_biases_hat = wts_bias_history_dict["second_history_biases"][i] / (1 - (optimizer_input_dict["beta2"] ** itr))

    lookahead_db = optimizer_input_dict["beta1"] * history_biases_hat + (((1-optimizer_input_dict["beta1"])/(1-(optimizer_input_dict["beta1"] ** itr))) * wts_bias_history_dict["db"][i])
    wts_bias_history_dict["biases"][i] = wts_bias_history_dict["biases"][i] - ((optimizer_input_dict["learning_rate"]/(np.sqrt(second_history_biases_hat) + optimizer_input_dict["epsilon"])) * (lookahead_db))
