import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepthermal.FFNN_model import FFNN, fit_FFNN, init_xavier
from deepthermal.validation import k_fold_CV_grid, get_rMSE, create_subdictionary_iterator

########
PATH_FIGURES = "/Users/alexander/git_repos/git_skole/deepthermal/figures"
PATH_TRAINING_DATA = "/Users/alexander/git_repos/git_skole/deepthermal/Task1/TrainingData.txt"
PATH_TESTING_POINTS = "/Users/alexander/git_repos/git_skole/deepthermal/Task1/TestingData.txt"

X_DATA_SCALE = 1e5
Y_DATA_SCALE = 5e3

model_params = {
    "input_dimension": [1],
    "output_dimension": [1],
    "n_hidden_layers": [2],
    "neurons": [20],
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

#########

# Data frame with data
df_train = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32)
df_test = pd.read_csv(PATH_TESTING_POINTS, dtype=np.float32)

# Load data
x_train = torch.tensor(df_train[["t"]].values) / X_DATA_SCALE
y_train = torch.tensor(df_train[["tf0"]].values) / Y_DATA_SCALE
x_test = torch.tensor(df_test[["t"]].values) / X_DATA_SCALE

# Number of training samples
n_samples = len(df_train.index)

model_params_iter = create_subdictionary_iterator(model_params)
training_params_iter = create_subdictionary_iterator(training_params)

models, rel_training_error, rel_val_errors = k_fold_CV_grid(Model=FFNN, model_param_iter=model_params_iter,
                                                            fit=fit_FFNN, training_param_iter=training_params_iter,
                                                            x=x_train, y=y_train, init=init_xavier, partial=False,
                                                            verbose=True)


## printing model errors
def print_model_errors(rel_val_errors):
    for i, rel_val_error_list in enumerate(rel_val_errors):
        avg_error = sum(rel_val_error_list) / len(rel_val_error_list)
        print(f"Model {i} val error: {avg_error * 100}%")

print_model_errors()

# plot visualization
def plot_models(model_number_list, models=models, plot_name="vis_model"):
    k = len(models[0])
    # num_models = len(model_number_list)
    for model_number in model_number_list:
        fig, axs = plt.subplots(1, k, figsize=(6 * k, 7))
        fig.suptitle(f"{plot_name}")

        for k_, model_k in enumerate(models[model_number]):
            axs[k_].scatter(x_train[:, 0], y_train[:, 0], label="tf0_train")
            # axs.scatter(x_train, y_train[:,1], label="ts0")

            axs[k_].plot(x_test, model_k(x_test).detach(), label=f"tf0_pred", lw=2, color="black", )

            axs[k_].set_xlabel("t")
            axs[k_].set_ylabel("T")
            axs[k_].legend()

        fig.savefig(f"{PATH_FIGURES}/{plot_name}_{model_number}.pdf")
        plt.close(fig)


def plot_model_history(model, model_name="0"):
    histfig, ax = plt.subplots()
    ax.grid(True, which="both", ls=":")
    ax.plot(np.arange(1, len(model.loss_history_train) + 1), model.loss_history_train, label="Training error history")
    ax.plot(np.arange(1, len(model.loss_history_val) + 1), model.loss_history_val, label="Validation error history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    histfig.suptitle(f"History for model {model_name}")
    histfig.savefig(f"{PATH_FIGURES}/model_{model_name}_history.pdf")
    plt.close(histfig)


plot_models([0], models)

chosen_network_properties = {
    "hidden_layers": [4],
    "neurons": 20,
    "regularization_exp": 2,
    "regularization_param": 1e-4,
    "batch_size": 32,
    "epochs": 1000,
    "optimizer": "ADAM",
    "init_weight_seed": 25
}
