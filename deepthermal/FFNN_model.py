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
        activation="relu",
        init=None,
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

        # init
        if init is not None:
            init(self)

    def forward(self, x):
        # The forward function performs the set of affine and
        # non-linear transformations defining the network
        # (see equation above)
        x = self.activation_(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation_(l(x))
        return self.output_layer(x)

    def __str__(self):
        return "FFNN"


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
    track_epoch=False,
    verbose=False,
    verbose_interval=100,
    learning_rate=None,
    init_weight_seed: int = None,
    lr_scheduler=None,
    loss_func=nn.MSELoss(),
    compute_loss: callable = compute_loss_torch,
    max_nan_steps=50,
    **kwargs
) -> tuple[callable, torch.Tensor, torch.Tensor]:
    if init is not None:
        init(model, init_weight_seed=init_weight_seed)

    training_set = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
    if learning_rate is None:
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
            max_iter=5000,
            max_eval=50000,
            history_size=100,
            line_search_fn="strong_wolfe",
        )
        max_nan_steps = 2
        verbose_interval = 1
    else:
        raise ValueError("Optimizer not recognized")

    # Learning Rate Scheduler

    if lr_scheduler is not None:
        scheduler = lr_scheduler(optimizer_)

    loss_history_train = list()
    loss_history_val = list()
    if track_epoch:
        loss_history_train = torch.zeros(num_epochs)
        loss_history_val = torch.zeros(num_epochs)

    nan_steps = 0
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

                # assumes that the expected loss is
                # not proportional to the length of training data
                if track_history and not track_epoch:
                    # track training loss
                    x_train, y_train = data[:]
                    train_loss = (
                        compute_loss(
                            loss_func=loss_func,
                            model=model,
                            x_train=x_train,
                            y_train=y_train,
                        )
                        .detach()
                        .item()
                    )
                    loss_history_train.append(train_loss)

                    # track validatoin loss
                    if data_val is not None and len(data_val) > 0 and track_history:
                        x_val, y_val = data_val[None]
                        validation_loss = (
                            compute_loss(
                                loss_func=loss_func,
                                model=model,
                                x_train=x_val,
                                y_train=y_val,
                            )
                            .detach()
                            .item()
                        )
                        loss_history_val.append(validation_loss)

                return loss

            optimizer_.step(closure=closure)

        if track_epoch or track_history or lr_scheduler:
            x_train, y_train = data[None]
            train_loss = compute_loss(
                loss_func=loss_func,
                model=model,
                x_train=x_train,
                y_train=y_train,
            ).detach()
            if track_history:
                # stop if nan output
                if torch.isnan(train_loss):
                    nan_steps += 1
                if epoch % 100 == 0:
                    nan_steps = 0

            if lr_scheduler is not None:
                scheduler.step(train_loss)

            if track_epoch:
                loss_history_train[epoch] = train_loss.item()
                if data_val is not None and len(data_val) > 0:
                    x_val, y_val = data_val[None]
                    validation_loss = (
                        compute_loss(
                            loss_func=loss_func,
                            model=model,
                            x_train=x_val,
                            y_train=y_val,
                        )
                        .detach()
                        .item()
                    )
                    loss_history_val = validation_loss.item

        if verbose and not epoch % verbose_interval and track_history:
            print("Training Loss: ", np.round(loss_history_train[-1], 8))
            if data_val is not None and len(data_val) > 0:
                print("Validation Loss: ", np.round(loss_history_val[-1], 8))

        if nan_steps > max_nan_steps:
            break

    if verbose and track_history and len(loss_history_train) > 0:
        print("Final training Loss: ", np.round(loss_history_train[-1], 8))
        if data_val is not None and len(data_val) > 0:
            print("Final validation Loss: ", np.round(loss_history_val[-1], 8))

    return model, torch.as_tensor(loss_history_train), torch.as_tensor(loss_history_val)


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
