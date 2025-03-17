import numpy as np
from keras.datasets import fashion_mnist, mnist
import numpy 

datasets = {"fashion_mnist": fashion_mnist, "mnist": mnist}

def load_data(dataset_name):
  (x_train, y_train), (x_test, y_test) = datasets[dataset_name].load_data()
  num_classes = len(np.unique(y_train))

  y_train = np.eye(num_classes)[y_train]
  y_test = np.eye(num_classes)[y_test]

  x_train = x_train.reshape(x_train.shape[0], -1)
  x_test = x_test.reshape(x_test.shape[0], -1)

  x_train = np.array(x_train/255, dtype=np.float64)
  y_train = np.array(y_train, dtype=np.float64)
  x_test = np.array(x_test/255, dtype=np.float64)
  y_test = np.array(y_test, dtype=np.float64)

  return x_train, y_train, x_test, y_test, num_classes
