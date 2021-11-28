import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# GLOBAL VARIABLES

larning_rates = {"ADAM": 0.001, "LBFGS": 0.1}

activations = {
    "LeakyReLU": nn.LeakyReLU,
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
}


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
        # The forward function performs the set of affine and
        # non-linear transformations defining the network
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
    return model


def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if "weight" or "bias" in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss


def compute_loss_torch(loss_func, model, y_train, x_train):
    y_pred = model(x_train)
    loss = loss_func(y_pred, y_train)
    return loss


def fit_FFNN(
    model: callable,
    data,
    num_epochs,
    batch_size,
    optimizer,
    init: callable = None,
    regularization_param=0,
    regularization_exp=2,
    data_val=None,
    track_history=True,
    verbose=False,
    learning_rate=None,
    init_weight_seed: int = None,
    loss_func=nn.MSELoss(),
    compute_loss: callable = compute_loss_torch,
    **kwargs
) -> tuple[callable, torch.Tensor, torch.Tensor]:
    if init is not None:
        init(model, init_weight_seed=init_weight_seed)

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
    elif optimizer == "strong_wolfe":
        optimizer_ = optim.LBFGS(
            model.parameters(),
            lr=learning_rate,
            max_iter=50000,
            max_eval=50000,
            history_size=100,
            line_search_fn="strong_wolfe",
        )
    else:
        raise ValueError("Optimizer not recognized")

    loss_history_train = torch.zeros((num_epochs))
    loss_history_val = torch.zeros((num_epochs))
    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose and not epoch % 100:
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
                loss_u = compute_loss(
                    loss_func=loss_func, model=model, x_train=x_train_, y_train=y_train_
                )
                loss_reg = regularization(model, regularization_exp)
                loss = loss_u + regularization_param * loss_reg
                loss.backward(retain_graph=True)

                return loss

            optimizer_.step(closure=closure)

            # Compute average training loss over batches for the current epoch
            if track_history:
                loss_train = compute_loss(
                    loss_func=loss_func, model=model, x_train=x_train_, y_train=y_train_
                )
                loss_history_train[epoch] += loss_train.item() / len(training_set)

        if data_val is not None and track_history:
            x_val, y_val = next(
                iter(DataLoader(data_val, batch_size=len(data_val), shuffle=False))
            )
            validation_loss = compute_loss(
                loss_func=loss_func, model=model, x_train=x_val, y_train=y_val
            ).detach()
            loss_history_val[epoch] = validation_loss

        if verbose and not epoch % 100 and track_history:
            print("Training Loss: ", np.round(loss_history_train[epoch].item(), 8))
            if data_val is not None:
                print("Validation Loss: ", np.round(validation_loss.item(), 8))

    if verbose and track_history:
        print("Final Training Loss: ", np.round(loss_history_train[-1].item(), 8))
        if data_val is not None:
            print("Final Validation Loss: ", np.round(loss_history_val[-1].item(), 8))

    return model, loss_history_train, loss_history_val


def get_trained_model(
    model_param,
    training_param,
    data,
    data_val=None,
    fit=fit_FFNN,
):
    # Xavier weight initialization
    model = model_param.pop("model")(**model_param)
    model, loss_history_train, loss_history_val = fit(
        model=model,
        data=data,
        data_val=data_val,
        **training_param,
        model_param=model_param,
    )
    return model, loss_history_train, loss_history_val


def get_scaled_model(model, x_center=0, x_scale=1, y_center=0, y_scale=1):
    def scaled_model(x):
        return model((x - x_center) / x_scale) * y_scale + y_center

    return scaled_model
