import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-wp", "--wandb_project", type=str, default="DA6401_Assignment1", 
                        help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str, default="nikhithaa-iit-madras", 
                        help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", 
                        help="Choose one among these datasets: ['mnist', 'fashion_mnist']")
    parser.add_argument("-e", "--epochs", type=int, default=10, 
                        help="Number of epochs to train neural network")
    parser.add_argument("-b", "--batch_size", type=int, default=16, 
                        help="Batch size used to train neural network")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy", 
                        help="Choose one among these loss functions: ['mean_squared_error', 'cross_entropy']")
    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="nadam", 
                        help="Choose one among these optimizers: ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, 
                        help="Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, 
                        help="Momentum used by momentum and nag optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.9, 
                        help="Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, 
                        help="Beta1 used by adam and nadam optimizers")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, 
                        help="Beta2 used by adam and nadam optimizers")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.00000001, 
                        help="Epsilon used by optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0005, 
                        help="Weight decay used by optimizers")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="Xavier", 
                        help="Choose one among these weight initialization methods: ['random', 'Xavier']")
    parser.add_argument("-nhl", "--num_layers", type=int, default=4, 
                        help="Number of hidden layers used in feedforward neural network")
    parser.add_argument("-sz", "--hidden_size", type=int, default=64, 
                        help="Number of hidden neurons in a feedforward layer")
    parser.add_argument("-a", "--activation", type=str, choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid", 
                        help="Choose one among these activation functions: ['identity', 'sigmoid', 'tanh', 'ReLU']")
    parser.add_argument("-cm", "--confusion_matrix", type=str, choices=["True", "False"], default="False",
                        help="Set true if confusion matrix to be logged")
    
    return parser.parse_args()
