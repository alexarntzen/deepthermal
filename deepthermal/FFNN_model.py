import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from typing import Union, List
from collections.abc import Callable

# GLOBAL VARIABLES

larning_rates = {"ADAM": 0.001, "LBFGS": 0.1, "strong_wolfe": 1}

activations = {
    "LeakyReLU": nn.LeakyReLU,
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
}


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


class FFNN(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        n_hidden_layers,
        neurons,
        activation: Union[str, callable] = "tanh",
        init=None,
        **kwargs,
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
        if isinstance(activation, str):
            self.activation_func = activations[self.activation]
        elif callable(activation):
            self.activation_func = activation()
        else:
            raise ValueError(f"Activation {activation} not recognized")

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)]
        )
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        # init
        if init is not None:
            init(self)

    def forward(self, x):
        # The forward function performs the set of affine and
        # non-linear transformations defining the network
        # (see equation above)
        x = self.activation_func(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation_func(l(x))
        return self.output_layer(x)

    def __str__(self):
        """reuturn name of class"""
        return type(self).__name__


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


def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if "weight" or "bias" in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss


def compute_loss_torch(
    model: nn.Module, data: Union[List, Dataset], loss_func: callable
) -> torch.Tensor:
    """default way"""
    x_train, y_train = data[:]
    y_pred = model(x_train)
    loss = loss_func(y_pred, y_train)
    print(loss)
    return loss


def fit_FFNN(
    model: nn.Module,
    data,
    num_epochs,
    batch_size,
    optimizer,
    init: callable = None,
    regularization_param=0,
    regularization_exp=2,
    data_val=None,
    track_history=True,
    track_epoch=True,
    verbose=False,
    learning_rate=None,
    init_weight_seed: int = None,
    lr_scheduler=None,
    loss_func=nn.MSELoss(),
    compute_loss: Callable[..., torch.Tensor] = compute_loss_torch,
    max_nan_steps=50,
    post_batch: Callable = None,
    post_epoch: Callable = None,
    **kwargs,
) -> tuple[Callable, np.array, np.array]:
    if init is not None:
        init(model, init_weight_seed=init_weight_seed)

    if learning_rate is None and not callable(optimizer):
        learning_rate = larning_rates[optimizer]
    # select optimizer
    if optimizer == "ADAM":
        optimizer_ = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "SGD":
        optimizer_ = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "LBFGS":
        optimizer_ = optim.LBFGS(
            model.parameters(),
            max_iter=1,
            max_eval=50000,
            tolerance_change=1.0 * np.finfo(float).eps,
            lr=learning_rate,
        )
    elif optimizer == "strong_wolfe":
        optimizer_ = optim.LBFGS(
            model.parameters(),
            lr=learning_rate,
            max_iter=100,
            max_eval=1000,
            history_size=200,
            line_search_fn="strong_wolfe",
        )
        max_nan_steps = 2
    elif callable(optimizer):
        optimizer_ = optimizer(model.parameters())
    else:
        raise ValueError("Optimizer not recognized")

    # Learning Rate Scheduler

    if lr_scheduler is not None:
        scheduler = lr_scheduler(optimizer_)

    loss_history_train = list()
    loss_history_val = list()
    if track_epoch:
        loss_history_train = np.zeros(num_epochs)
        loss_history_val = np.zeros(num_epochs)

    nan_steps = 0
    # Loop over epochs
    epohcs_tqdm = tqdm(
        range(num_epochs), desc="Epoch: ", disable=(not verbose), leave=False
    )
    for epoch in epohcs_tqdm:
        training_set = DataLoader(
            data, batch_size=batch_size, shuffle=True, drop_last=True
        )
        # try one epoch, break if interupted:
        try:
            for j, data_sample in enumerate(training_set):

                def closure():
                    # zero the parameter gradients
                    optimizer_.zero_grad()
                    # forward + backward + optimize
                    loss_u = compute_loss(
                        model=model,
                        data=data_sample,
                        loss_func=loss_func,
                    )
                    loss_reg = regularization(model, regularization_exp)
                    loss = loss_u + regularization_param * loss_reg
                    loss.backward()

                    return loss

                optimizer_.step(closure=closure)

                if post_batch is not None:
                    post_batch(model=model, data=data)

                # track after each step if not track epoch
                # assumes that the expected loss is
                # not proportional to the length of training data
                if track_history and not track_epoch:
                    # track training loss
                    train_loss = compute_loss(
                        model=model, data=data, loss_func=loss_func
                    ).item()
                    loss_history_train.append(train_loss)

                    # track validation loss
                    if data_val is not None and len(data_val) > 0 and track_history:
                        validation_loss = compute_loss(
                            model=model, data=data_val, loss_func=loss_func
                        ).item()
                        loss_history_val.append(validation_loss)

            if post_epoch is not None:
                post_epoch(model=model, data=data)

            if track_epoch or track_history or lr_scheduler:
                train_loss = compute_loss(
                    model=model, data=data, loss_func=loss_func
                ).item()
                if track_history:
                    # stop if nan output
                    if np.isnan(train_loss):
                        nan_steps += 1
                    if epoch % 100 == 0:
                        nan_steps = 0

                if lr_scheduler is not None:
                    scheduler.step(train_loss)

                if track_epoch:
                    loss_history_train[epoch] = train_loss
                    if data_val is not None and len(data_val) > 0:
                        validation_loss = compute_loss(
                            model=model,
                            data=data_val,
                            loss_func=loss_func,
                        ).item()
                        loss_history_val[epoch] = validation_loss
            if verbose and track_history:
                print_iter = epoch if track_epoch else -1
                if data_val is not None and len(data_val) > 0:
                    epohcs_tqdm.set_postfix(
                        loss=loss_history_train[print_iter],
                        val_loss=loss_history_val[print_iter],
                    )
                else:
                    epohcs_tqdm.set_postfix(
                        loss=loss_history_train[print_iter],
                    )
            if nan_steps > max_nan_steps:
                break

        except KeyboardInterrupt:
            print("Interrupted breaking")
            break

    if verbose and track_history and len(loss_history_train) > 0:
        print("\nFinal training Loss: ", np.round(loss_history_train[-1], 8))
        if data_val is not None and len(data_val) > 0:
            print("Final validation Loss: ", np.round(loss_history_val[-1], 8))

    return model, np.array(loss_history_train), np.array(loss_history_val)


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
