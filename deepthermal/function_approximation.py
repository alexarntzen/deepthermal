import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

from deepthermal.nn_model import get_trained_nn_model, create_subdictionary_iterator

########

PATH_TRAINING_DATA = "/Users/alexander/Kodeprosjekter/deepthermal/Task1/TrainingData.txt"
PATH_FIGURES = "/Users/alexander/Kodeprosjekter/deepthermal/figures/"

SAMPLING_SEED = 78


#########

def method1():
    return True


# Data frame with data
df = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32)
x_train = torch.tensor(df[["t"]].values) * 1e-5
y_train = torch.tensor(df[["tf0"]].values) * 5e-3
x_axis = torch.reshape(torch.linspace(torch.min(x_train), torch.max(x_train), 1000), (1000, 1))
# Random Seed for dataset generation
torch.manual_seed(SAMPLING_SEED)

# Number of training samples
n_samples = len(df.index)

model_params = {
    "input_dimension": [1],
    "output_dimension": [1],
    "n_hidden_layers": [2],
    "neurons": [n_samples // 8],
    "activation": ["relu"],
    "init_weight_seed": [20]
}
training_params = {
    "num_epochs": [1000],
    "batch_size": [20],
    "regularization_exp": [2],
    "regularization_param": [1e-4],
    "optimizer": ["ADAM"],
}

train_err_conf = list()
val_err_conf = list()
# test_err_conf = list()

settings = create_subdictionary_iterator(model_params)

valfig = plt.figure()
plt.scatter(x_train, y_train, label="tf0")
# plt.scatter(x_train, y_train[:,1], label="ts0")

plt.xlabel("T")
plt.ylabel("t")
plt.title(f"Values")
model = None
for set_num, setup_properties in enumerate(settings):
    print("###################################", set_num, "###################################")

    relative_error_train_, relative_error_val_, training_history, model = run_configuration_with_cv(
        conf_dict=setup_properties,
        x=x_train,
        y=y_train
    )
    train_err_conf.append(relative_error_train_)
    val_err_conf.append(relative_error_val_)

    histfig = plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, setup_properties["epochs"] + 1), training_history[0], label="Training error history")
    plt.plot(np.arange(1, setup_properties["epochs"] + 1), training_history[1], label="Validation error history")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"History for model {set_num}")
    plt.legend()
    plt.savefig(f"{PATH_FIGURES}model_{set_num}_history.pdf", )
    plt.close(histfig)

    plt.plot(x_axis, model(x_axis).detach(), color="orange", label=f"model_{set_num}")

plt.legend()
plt.savefig(f"{PATH_FIGURES}model_plot_0.pdf", )
plt.close(valfig)

train_err_conf = np.array(train_err_conf)
val_err_conf = np.array(val_err_conf)

chosen_network_properties = {
    "hidden_layers": [2, 4],
    "neurons": [5, 20],
    "regularization_exp": [2],
    "regularization_param": [1e-3],
    "batch_size": [n_samples],
    "epochs": [1000],
    "optimizer": ["ADAM"],
    "init_weight_seed": [5]
}
