import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepthermal.FFNN_model import FFNN, fit_FFNN, init_xavier
from deepthermal.validation import k_fold_CV_grid, create_subdictionary_iterator

########
PATH_FIGURES = "/Users/alexander/git_repos/git_skole/deepthermal/figures"
PATH_TRAINING_DATA = "/Users/alexander/git_repos/git_skole/deepthermal/Task1/TrainingData.txt"
PATH_TESTING_POINTS = "/Users/alexander/git_repos/git_skole/deepthermal/Task1/TestingData.txt"

SET_NAME = "2d_wide_relu_1"
MODEL_LIST = [0, ]

model_params = {
    "input_dimension": [1],
    "output_dimension": [2],
    "n_hidden_layers": [5],
    "neurons": [200],
    "activation": ["relu"],
    "init_weight_seed": [25]
}
training_params = {
    "num_epochs": [2000],
    "batch_size": [265 // 8],
    "regularization_exp": [2],
    "regularization_param": [1e-4],
    "optimizer": ["ADAM"],
}

#########

# Data frame with data
df_train = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32)
df_test = pd.read_csv(PATH_TESTING_POINTS, dtype=np.float32)

# Load data
x_train_ = torch.tensor(df_train[["t"]].values)
y_train = torch.tensor(df_train[["tf0","ts0"]].values)
x_test_ = torch.tensor(df_test[["t"]].values)

# Normalize values
X_TRAIN_MEAN = torch.mean(x_train_)
X_TRAIN_STD = torch.std(x_train_)
x_train = (x_train_ - X_TRAIN_MEAN) / X_TRAIN_STD
x_test = (x_test_ - X_TRAIN_MEAN) / X_TRAIN_STD

# Number of training samples
n_samples = len(df_train.index)

model_params_iter = create_subdictionary_iterator(model_params)
training_params_iter = create_subdictionary_iterator(training_params)

models, rel_training_error, rel_val_errors = k_fold_CV_grid(Model=FFNN, model_param_iter=model_params_iter,
                                                            fit=fit_FFNN, training_param_iter=training_params_iter,
                                                            x=x_train, y=y_train, init=init_xavier, partial=True,
                                                            verbose=True)


## printing model errors
def print_model_errors(rel_val_errors):
    for i, rel_val_error_list in enumerate(rel_val_errors):
        avg_error = sum(rel_val_error_list) / len(rel_val_error_list)
        print(f"Model {i} validation error: {avg_error * 100}%")


def get_disc_str(model):
    params = {"activation": model.activation, "n_hidden_layers": model.n_hidden_layers, "neurons": model.neurons,
              "epochs": len(model.loss_history_train)}
    return str(params)


# plot visualization
def plot_models(model_number_list, models=models, plot_name="vis_model"):
    k = len(models[0])
    # num_models = len(model_number_list)
    for model_number in model_number_list:
        fig, axs = plt.subplots(1, k, figsize=(8 * k, 6))
        fig.suptitle(f"Model: {get_disc_str(models[model_number][0])}")

        for k_, ax in enumerate(fig.axes):
            ax.scatter(x_train[:, 0], y_train[:, 0], label="tf0_train")
            axs.scatter(x_train, y_train[:,1], label="ts0_train")

            ax.plot(x_test, models[model_number][k_](x_test)[:,0].detach(), label=f"tf0_pred", lw=2, ls="-.",
                    color="black", )
            ax.plot(x_test, models[model_number][k_](x_test)[:, 1].detach(), label=f"tf0_pred", lw=2, ls="-.",
                    color="green", )

            ax.set_xlabel("t")
            ax.set_ylabel("T")
            ax.legend()

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
    histfig.suptitle(f"History, model: {get_disc_str(model)}")
    histfig.savefig(f"{PATH_FIGURES}/history_{model_name}.pdf")
    plt.close(histfig)


print_model_errors(rel_val_errors)
plot_models(MODEL_LIST, models, "result_" + SET_NAME)
for i in MODEL_LIST:
    plot_model_history(model=models[i][0], model_name=(SET_NAME + f"{i}"))

chosen_network_properties = {
    "hidden_layers": 4,
    "neurons": 20,
    "regularization_exp": 2,
    "regularization_param": 1e-4,
    "batch_size": 32,
    "epochs": 1000,
    "optimizer": "ADAM",
    "init_weight_seed": 25
}
