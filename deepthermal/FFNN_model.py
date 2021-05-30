import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# GLOBAL VARIABLES

larning_rates = {"ADAM": 0.001, "LBFGS": 0.1}

activations = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}


class FFNN(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        n_hidden_layers,
        neurons,
        activation="relu",
        **kwargs
    ):
        super(FFNN, self).__init__()

        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = activation

        self.activation_ = activations[self.activation]

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers)]
        )
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        # (see equation above)
        x = self.activation_(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation_(l(x))
        return self.output_layer(x)


def NeuralNet_Seq(input_dimension, output_dimension, n_hidden_layers, neurons):
    modules = list()
    modules.append(nn.Linear(input_dimension, neurons))
    modules.append(nn.Tanh())
    for _ in range(n_hidden_layers):
        modules.append(nn.Linear(neurons, neurons))
        modules.append(nn.Tanh())
    modules.append(nn.Linear(neurons, output_dimension))
    model = nn.Sequential(*modules)
    return model


def init_xavier(model, init_weight_seed=None, **kwargs):
    if init_weight_seed is not None:
        torch.manual_seed(init_weight_seed)

    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain(model.activation)
            # torch.nn.init.xavier_uniform_(m.weight, gain=g)
            torch.nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)

    model.apply(init_weights)


def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if "weight" or "bias" in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss


def fit_FFNN(
    model,
    data,
    num_epochs,
    batch_size,
    optimizer,
    p=2,
    regularization_param=0,
    regularization_exp=2,
    data_val=None,
    track_history=True,
    verbose=False,
    learning_rate=None,
    **kwargs
):
    training_set = DataLoader(data, batch_size=batch_size, shuffle=True)
    if learning_rate is None:
        learning_rate = larning_rates[optimizer]
    # select optimizer
    if optimizer == "ADAM":
        optimizer_ = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "LBFGS":
        optimizer_ = optim.LBFGS(
            model.parameters(),
            lr=learning_rate,
            max_iter=1,
            max_eval=50000,
            tolerance_change=1.0 * np.finfo(float).eps,
        )
    else:
        raise ValueError("Optimizer not recognized")

    loss_history_train = np.zeros((num_epochs))
    loss_history_val = np.zeros((num_epochs))
    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose:
            print(
                "################################ ",
                epoch,
                " ################################",
            )

        # Loop over batches
        for j, (x_train_, y_train_) in enumerate(training_set):

            def closure():
                # zero the parameter gradients
                optimizer_.zero_grad()
                # forward + backward + optimize
                y_pred_ = model(x_train_)
                loss_u = torch.mean((y_pred_ - y_train_) ** p)
                loss_reg = regularization(model, regularization_exp)
                loss = loss_u + regularization_param * loss_reg
                loss.backward()

                # Compute average training loss over batches for the current epoch
                if track_history:
                    loss_history_train[epoch] += loss.item() / len(training_set)
                return loss

            optimizer_.step(closure=closure)

        if data_val:
            x_val, y_val = next(
                iter(DataLoader(data_val, batch_size=len(data_val), shuffle=False))
            )
        # record validation loss for history
        if data_val is not None and track_history:
            y_val_pred_ = model(x_val)
            validation_loss = torch.mean((y_val_pred_ - y_val) ** p).item()
            loss_history_val[epoch] = validation_loss

        if verbose and track_history:
            print("Training Loss: ", np.round(loss_history_train[-1], 8))
            if data_val is not None:
                print("Validation Loss: ", np.round(validation_loss, 8))

    if verbose and track_history:
        print("Final Training Loss: ", np.round(loss_history_train[-1], 8))
        if data_val is not None:
            print("Final Validation Loss: ", np.round(loss_history_val[-1], 8))

    return loss_history_train, loss_history_val


def get_trained_nn_model(model_param, training_param, data, data_val=None):
    nn_model = FFNN(**model_param)
    # Xavier weight initialization
    init_xavier(nn_model, **training_param)

    loss_history_train, loss_history_val = fit_FFNN(
        nn_model, data, data_val=data_val, **training_param
    )
    return nn_model, loss_history_train, loss_history_val
