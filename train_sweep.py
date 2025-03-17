from neural_network import *
from dataset_load import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sweep_configuration = {
    "method": "bayes",
    "name" : "train_sweep",
    "metric": {"name": "validation_accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [1e-3, 1e-4]},
        "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop",  "adam", "nadam"]},
        "num_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [32, 64, 128]},
        "batch_size": {"values": [16, 32, 64]},
        "activation": {"values": ["sigmoid", "tanh", "ReLU"]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "weight_init": {"values": ["random", "Xavier"]},
        "epochs": {"values": [10, 5]},
        "loss": {"values": ["cross_entropy"]},
        "momentum": {"values": [0.9]},
        "beta": {"values": [0.9]},
        "beta1": {"values":[0.9]},
        "beta2": {"values": [0.999]},
        "epsilon": {"values": [1e-8]},
        "dataset": {"values":["fashion_mnist"]}
    }
}

def calculateAccuracy(y_true, y_pred):
  y_pred_labels = np.argmax(y_pred, axis=1)
  y_true_labels = np.argmax(y_true, axis=1)
  accuracy = np.mean(y_pred_labels == y_true_labels)
  return accuracy*100

def trainNeuralNetwork_sweep():
  wandb.init(mode="online")
  args = wandb.config
  x_train, y_train, x_test, y_test, num_classes = load_data(args["dataset"])
  input_size = len(x_train[0])
  output_size = num_classes
  n_hiddenLayers = args["num_layers"]
  n_neuronsPerLayer = args["hidden_size"]
  activationFun = args["activation"]
  weight_init = args["weight_init"]
  batch_size = args["batch_size"]
  lossFunc = args["loss"]
  optimizer = args["optimizer"]
  learning_rate = args["learning_rate"]
  momentum = args["momentum"]
  beta = args["beta"]
  beta1 = args["beta1"]
  beta2 = args["beta2"]
  epsilon = args["epsilon"]
  weight_decay = args["weight_decay"]
  epochs = args["epochs"]

  wandb.run.name = f"{optimizer}_{activationFun}_{n_hiddenLayers}_{n_neuronsPerLayer}_{epochs}_{weight_init}"

  # paste all above paramters as fun params
  fnn = FeedForwardNeuralNetwork(input_size, output_size, n_hiddenLayers, n_neuronsPerLayer,
                                 activationFun, weight_init, batch_size, lossFunc,
                                 optimizer, learning_rate, momentum,
                                 beta, beta1, beta2,
                                 epsilon, weight_decay, epochs)

  x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
  num_batches = math.ceil(len(x_train)/batch_size)

  for epochNum in range(epochs):
    for batchNum in range(num_batches):
      start_idx = batchNum * batch_size
      end_idx = start_idx + batch_size

      x_batch = x_train[start_idx:end_idx]
      y_batch = y_train[start_idx:end_idx]

      # Forward Propagation
      a_pre_activation, h_post_activation = fnn.forwardPropagation(x_batch)
      y_pred_batch = h_post_activation[-1]

      # Back Propagation
      grad_w, grad_b = fnn.backwardPropagation(a_pre_activation, h_post_activation, y_batch, y_pred_batch)

      # Update weights
      itr = epochNum * num_batches + batchNum + 1
      fnn.updateWeights(grad_w, grad_b, itr)

    # Validation accuracy
    _, h_validation = fnn.forwardPropagation(x_validation, isValidation=True)
    y_pred_validation = h_validation[-1]
    validation_accuracy = calculateAccuracy(y_validation, y_pred_validation)
    wandb.run.summary["metric_name"] = validation_accuracy


    # Train accuracy
    _, h_train = fnn.forwardPropagation(x_train, isValidation=True)
    y_pred_train = h_train[-1]
    train_accuracy = calculateAccuracy(y_train, y_pred_train)

    wandb.log({
        "epoch": epochNum + 1,
        "validation_loss": np.mean(fnn.lossFunc(y_validation, y_pred_validation)),
        "validation_accuracy": validation_accuracy,
        "train_loss": np.mean(fnn.lossFunc(y_train, y_pred_train)),
        "train_accuracy": train_accuracy
        },commit=True)

  # Test accuracy
  _,h_test = fnn.forwardPropagation(x_test, isValidation=True)
  y_pred_test = h_test[-1]
  test_accuracy = calculateAccuracy(y_test, y_pred_test)
  wandb.log({ "test_accuracy": test_accuracy,
             "test_loss": np.mean(fnn.lossFunc(y_test, y_pred_test))})

  # Confusion matrix
  class_names = []
  if(args["isConfusionMatrix"] == "True"):
      if(args["dataset"] == "fashion_mnist"):
          class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress","Coat","Sandal", "Shirt", "Sneaker","Bag","Ankle boot"]
      elif(args["dataset"] == "mnist"):
          class_names = [str(i) for i in range(10)]
          
  confusion_mat = confusion_matrix(y_pred_test.argmax(axis=1), y_test.argmax(axis=1))

  # plot
  plt.figure(figsize=(8,8))
  sns.heatmap(confusion_mat, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Greens")
  plt.xlabel("y_true")
  plt.ylabel("y_pred")
  plt.title("Confusion Matrix")

  wandb.log({"confusion_matrix": wandb.Image(plt)})
  plt.close()

  wandb.finish()

wandb.login()
wandb_id = wandb.sweep(sweep_configuration, project="DA6401_Assignment1")
wandb.agent(wandb_id, function=trainNeuralNetwork_sweep)
