import libraries
from activation_functions import *
from loss_functions import *
from optimizers import *
from derivatives import *
import numpy as np
import copy
import math

class FeedForwardNeuralNetwork:
  # class variables
  optimizersMap = {"sgd": sgd, "momentum": momentumGradientDescent, "nag": nag, "rmsprop": rmsProp, "adam": adam, "nadam": nadam}
  lossFunctionsMap = {"mean_squared_error": mean_squared_error, "cross_entropy" : cross_entropy_loss}
  activationFunctionsMap = {"identity":identity, "sigmoid":sigmoid, "tanh":tanh, "ReLU":relu, "softmax": softmax}
  derivatesFuncMap = {"mean_squared_error": mean_squared_error_derivative, "cross_entropy_loss": cross_entropy_loss_derivative, "identity": identity_derivative,
                      "sigmoid": sigmoid_derivative, "tanh": tanh_derivative, "relu": relu_derivative, "softmax": softmax_derivative}

  def __init__(self,
               input_size=784, output_size=10,
               n_hiddenLayers=3, n_neuronsPerLayer=32,
               activationFun="sigmoid",
               weight_init="random",
               batch_size=64,
               lossFunc="cross_entropy",
               optimizer="adam",
               learning_rate=0.001,
               momentum=0.5,
               beta=0.9, beta1=0.9, beta2=0.99,
               epsilon=1e-8, weight_decay=0.01,
               epochs=10):

    # Inialtization parameters
    self.input_size = input_size  # no of features
    self.output_size = output_size
    self.n_hiddenLayers = n_hiddenLayers
    self.n_neuronsPerLayer = n_neuronsPerLayer
    self.weight_init = weight_init
    self.epochs = epochs

    self.activationFun = FeedForwardNeuralNetwork.activationFunctionsMap[activationFun]
    self.lossFunc = FeedForwardNeuralNetwork.lossFunctionsMap[lossFunc]
    self.optimizer = FeedForwardNeuralNetwork.optimizersMap[optimizer]

    # paramters required for optimizers
    self.batch_size = batch_size
    self.isLookAhead = False;

    if(optimizer == "nag"):
      self.isLookAhead = True;

    # add these parameters as dict
    self.optimizer_input_dict = { "learning_rate" : learning_rate,
                                  "momentum" : momentum,                  # used by momentumGD
                                  "beta" : beta,                          # used by rmsprop
                                  "beta1" : beta1,                        # used by adam & nadam
                                  "beta2" : beta2,                        # used by adam & nadam
                                  "epsilon" : epsilon,
                                  "weight_decay" : weight_decay,
                                  "n_hiddenLayers": n_hiddenLayers}

    # weights and biases matrices
    self.weights = []
    self.biases = []
    self.lookAheadWeights = []
    self.lookAheadBiases = []

    self.wts_bias_history_dict = {"weights": self.weights, "biases": self.biases,
                                  "history_weights": [np.zeros(1) for _ in range(self.n_hiddenLayers+1)],         # these will be modified before their first use (dimensions of each values will also be changed)
                                  "history_biases": [np.zeros(1) for _ in range(self.n_hiddenLayers+1)],
                                  "second_history_weights": [np.zeros(1) for _ in range(self.n_hiddenLayers+1)],
                                  "second_history_biases": [np.zeros(1) for _ in range(self.n_hiddenLayers+1)],
                                  "dw": [np.empty(1) for _ in range(self.n_hiddenLayers+1)],
                                  "dh": [np.empty(1) for _ in range(self.n_hiddenLayers+1)]}

    self.initializeWeightsAndBiases()
    self.wts_bias_history_dict["second_history_weights"] = copy.deepcopy(self.wts_bias_history_dict["history_weights"])
    self.wts_bias_history_dict["second_history_biases"] = copy.deepcopy(self.wts_bias_history_dict["history_biases"])

    # pre-activation(a) and post-activation(h) values
    self.a = []
    self.h = []

  '''
    Weights,Biases initialization based on weight_init parameter

    weights[0]: input layer to first hidden layer  : input_size x n_neuronsPerLayer
    weights[1]: first hidden layer to second hidden layer : n_neuronsPerLayer x n_neuronsPerLayer
    ...
    weights[n_hiddenLayers]: last hidden layer to output layer : n_neuronsPerLayer x output_size

    biases[i] : bias for ith layer : 1 x n_neuronsPerLayer   (i:0 to n_hiddenLayers-1)
    biases[n_hiddenLayers]: 1 x output_size
  '''
  def initializeWeightsAndBiases(self):
    # biases for both types
    for i in range(self.n_hiddenLayers):
      self.biases.append(np.zeros(self.n_neuronsPerLayer))
      self.wts_bias_history_dict["history_biases"][i] = np.zeros(self.n_neuronsPerLayer)

    self.biases.append(np.zeros(self.output_size))   # biases[n_hiddenLayers]
    self.wts_bias_history_dict["history_biases"][self.n_hiddenLayers] = np.zeros(self.output_size)

    if(self.weight_init == "random"):   # Random Normal
      # weights[0]
      self.weights.append(np.random.randn(self.input_size, self.n_neuronsPerLayer))
      self.wts_bias_history_dict["history_weights"][0] = np.zeros((self.input_size, self.n_neuronsPerLayer))

      # weights[1] -> weights[n_hiddenLayers-1]
      for i in range(self.n_hiddenLayers-1):
        self.weights.append(np.random.randn(self.n_neuronsPerLayer, self.n_neuronsPerLayer))
        self.wts_bias_history_dict["history_weights"][i+1] = np.zeros((self.n_neuronsPerLayer, self.n_neuronsPerLayer))

      # weights[n_hiddenLayers]
      self.weights.append(np.random.randn(self.n_neuronsPerLayer, self.output_size))
      self.wts_bias_history_dict["history_weights"][self.n_hiddenLayers] = np.zeros((self.n_neuronsPerLayer, self.output_size))

    elif(self.weight_init == "Xavier"):   # Xavier Normal: mean = 0, variance = 2/(n_input + n_output)
      # weights[0]
      self.weights.append(np.random.normal(loc=0.0, scale=np.sqrt(2/(self.input_size + self.n_neuronsPerLayer)), size=(self.input_size, self.n_neuronsPerLayer)))
      self.wts_bias_history_dict["history_weights"][0] = np.zeros((self.input_size, self.n_neuronsPerLayer))


      for i in range(self.n_hiddenLayers-1):
        self.weights.append(np.random.normal(loc=0.0, scale=np.sqrt(2/(self.n_neuronsPerLayer + self.n_neuronsPerLayer)), size=(self.n_neuronsPerLayer, self.n_neuronsPerLayer)))
        self.wts_bias_history_dict["history_weights"][i+1] = np.zeros((self.n_neuronsPerLayer, self.n_neuronsPerLayer))


      self.weights.append(np.random.normal(loc=0.0, scale=np.sqrt(2/(self.n_neuronsPerLayer + self.output_size)), size=(self.n_neuronsPerLayer, self.output_size)))
      self.wts_bias_history_dict["history_weights"][self.n_hiddenLayers] = np.zeros((self.n_neuronsPerLayer, self.output_size))

  '''
    Forward propagation through the neural network. (for batch)
    Instead of doing one input at a time, this function handles it for a batch using respective sized matrices

    x_batch: B x n where B - batch size, n- no of features = input_size
    x_batch is assumbed to be numpy array when given as input
  '''
  def forwardPropagation(self, x_batch, isValidation=False):
    a_pre_activation = []
    h_post_activation = []

    # considering a0,h0 as X values as a1: first layer  (it is calculated from x values)
    a_pre_activation.append(x_batch)
    h_post_activation.append(x_batch)

    wt = []
    b = []

    if(self.isLookAhead and not isValidation):
      for i in range(self.n_hiddenLayers+1):
        wt.append(self.weights[i] - (self.optimizer_input_dict["momentum"] * self.wts_bias_history_dict["history_weights"][i]))
        b.append(self.biases[i] - (self.optimizer_input_dict["momentum"] * self.wts_bias_history_dict["history_biases"][i]))

      self.lookAheadWeights = wt
      self.lookAheadBiases = b
    else:
      wt = copy.deepcopy(self.weights)
      b = copy.deepcopy(self.biases)

    # Except last layer since activation function could be different
    for i in range(self.n_hiddenLayers):
      # ai: B x n_neuronsPerLayer, biases[i]: 1 x n_neuronsPerLayer (it will be broadcasted while adding)
      ai = np.matmul(h_post_activation[-1], wt[i]) + b[i]
      hi = self.activationFun(ai)

      a_pre_activation.append(ai)
      h_post_activation.append(hi)

    # aL: last layer (activation function is softmax)
    aL = np.matmul(h_post_activation[-1], wt[self.n_hiddenLayers]) + b[self.n_hiddenLayers]
    hL = softmax(aL)   # y_batch

    a_pre_activation.append(aL)
    h_post_activation.append(hL)

    return a_pre_activation, h_post_activation

  '''
    Backward propagation through the neural network. (for batch)
  '''
  def backwardPropagation(self, a_pre_activation, h_post_activation, y_batch, y_pred_batch):
    grad_w = []
    grad_b = []
    grad_a = []
    grad_h = []

    wt = []
    b = []
    if(self.isLookAhead):
        wt = self.lookAheadWeights
        b = self.lookAheadBiases 
    else:
        wt = copy.deepcopy(self.weights)
        b = copy.deepcopy(self.biases)  

    # Output gradient (wrt aL)
    grad_hL = self.derivatesFuncMap[self.lossFunc.__name__](y_batch, y_pred_batch)
    grad_h.append(grad_hL)

    if(self.lossFunc.__name__ == "cross_entropy_loss"):
      grad_aL = y_pred_batch - y_batch    # just to reduce computation of jacobian matrix
      grad_a.append(grad_aL)
    else:
      grad_aL_list = []
      # softmax derivatives of each input is a matrix of size output_size x output_size, we need to perform matrix_mul for each input of batch
      for i in range(y_batch.shape[0]):   # self.batch_size = y_batch.shape[0] but better to take y_batch.shape[0] since last batch inputs can have less
        grad_aL_inp_i = grad_hL[i] @ softmax_derivative(y_pred_batch[i])
        grad_aL_list.append(grad_aL_inp_i)

      grad_aL = np.array(grad_aL_list)
      grad_aL = grad_aL / y_batch.shape[0]
      grad_a.append(grad_aL)                    # aL contains (aL) values of all inputs in the batch

    # Hidden layers
    for k in range(self.n_hiddenLayers, -1, -1):
      # gradients w.r.t parameters
      # wk
      grad_wk = np.zeros_like(wt[k])    # will be equal to sum across

      for inpNum in range(y_batch.shape[0]):
        grad_wk_inp_num = np.matmul(h_post_activation[k][inpNum].reshape(-1,1), grad_a[-1][inpNum].reshape(1,-1))
        grad_wk += grad_wk_inp_num
      grad_w.append(grad_wk)                   # contains sum across all batches

      # bk
      grad_bk = np.zeros_like(self.biases[k])
      for inpNum in range(y_batch.shape[0]):
        grad_bk += grad_a[-1][inpNum]
      grad_b.append(grad_bk)                     # contains sum across all batches

      if(k > 0):
        # gradients w.r.t layer below
        grad_hk_1 = grad_a[-1] @ wt[k].T
        grad_h.append(grad_hk_1)

        # gradients w.r.t layer below (pre-activation)
        grad_ak_1 = grad_hk_1 * self.derivatesFuncMap[self.activationFun.__name__](a_pre_activation[k])
        grad_a.append(grad_ak_1)

    grad_w = grad_w[::-1]
    grad_b = grad_b[::-1]

    for i in range(self.n_hiddenLayers):
        grad_w[i] = grad_w[i] + (self.optimizer_input_dict["weight_decay"] * wt[i])

    return grad_w, grad_b

  def updateWeights(self, grad_w, grad_b, itr):
    grad_w = [np.clip(dw, -10,10) for dw in grad_w]
    grad_h = [np.clip(db, -10,10) for db in grad_b]
    self.wts_bias_history_dict["dw"] = grad_w
    self.wts_bias_history_dict["db"] = grad_b
    self.optimizer(self.optimizer_input_dict, self.wts_bias_history_dict, itr)
