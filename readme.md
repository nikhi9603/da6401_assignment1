# Feedforward Neural Network

This repository contains an implementation of feedforward neural network supporting multiple optimizers, activation functions and loss functions. 
Implementation can be found in python notebook and also in different files across directory as mentioned below.

## Code Organization
- `activation_functions.py` - Implements different activation functions.
- `argument_parser.py` - Handles command-line argument parsing.
- `dataset_load.py` - Loads and processes datasets for training (MNIST, Fashion-MNIST).
- `derivatives.py` - Contains derivatives of all activation and loss functions
- `libraries.py` - Contains all library imports.
- `loss_functions.py` - Implements different loss functions.
- `neural_network.py` - Defines the feed forward neural network architecture. Contains both forward and back propagation.
- `optimizers.py` - Implements different optimization algorithms.
- `train.py` - Main script for training the neural network. 
- `train_sweep.py` - Contains script for training the neural network as a sweep unlike train.py which supports only one run
- `Feedforward_Neural_Network.ipynb` - Jupyter Notebook to run training interactively.

## Running the Training Script
To train the neural network, run:

```sh
python train.py --wandb_entity myname --wandb_project myprojectname
```
Along with the wandb entity, project different arguments are supported from the command line as shown below. <br>
(For train_sweep.py, one has to set the sweet configuration manually to train the network)

### Supported Arguments for train.py

| Name | Default Value | Description |
|------|--------------|-------------|
| `-wp`, `--wandb_project` | `myprojectname` | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | `myname` | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | `fashion_mnist` | Choices: [`mnist`, `fashion_mnist`] |
| `-e`, `--epochs` | `1` | Number of epochs to train neural network. |
| `-b`, `--batch_size` | `4` | Batch size used to train neural network. |
| `-l`, `--loss` | `cross_entropy` | Choices: [`mean_squared_error`, `cross_entropy`] |
| `-o`, `--optimizer` | `sgd` | Choices: [`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`] |
| `-lr`, `--learning_rate` | `0.1` | Learning rate used to optimize model parameters. |
| `-m`, `--momentum` | `0.5` | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | `0.5` | Beta used by rmsprop optimizer. |
| `-beta1`, `--beta1` | `0.5` | Beta1 used by adam and nadam optimizers. |
| `-beta2`, `--beta2` | `0.5` | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | `0.000001` | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | `0.0` | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | `random` | Choices: [`random`, `Xavier`] |
| `-nhl`, `--num_layers` | `1` | Number of hidden layers used in feedforward neural network. |
| `-sz`, `--hidden_size` | `4` | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | `sigmoid` | Choices: [`identity`, `sigmoid`, `tanh`, `ReLU`] |
| `-cm`, `--confusion_matrix`| `False` | Set it to true if confusion matrix has to be logged while running. Choices : [`True`, `False`] |

## Running with Jupyter Notebook
Each section is mentioned seperately in the notebook. One can train the network by following through the sections in the notebook till training.
Ipython magic writing is commented in each block. One can make of use of them if files has to be modified in the directory directly.

## Link to Github
Github: https://github.com/nikhi9603/da6401_assignment1

## Link to Wandb report 
Wandb Report: https://api.wandb.ai/links/nikhithaa-iit-madras/pajt1ied