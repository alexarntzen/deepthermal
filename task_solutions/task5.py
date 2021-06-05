# might be linear
# takes about an hour for 1000 points
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qmcpy.discrete_distribution.sobol.sobol as sobol

from deepthermal.FFNN_model import FFNN, fit_FFNN, init_xavier
from deepthermal.validation import (
    k_fold_cv_grid,
    create_subdictionary_iterator,
)
from deepthermal.plotting import plot_result, plot_model_scatter
from task_solutions.task5_model_params import MODEL_PARAMS_CF, TRAINING_PARAMS_CF
from deepthermal.optimization import argmin

# some notes, use lbfgs?

# Path data
########
PATH_FIGURES = "figures/task5"
PATH_TRAINING_DATA = "Task5/TrainingData.txt"
PATH_SUBMISSION = "alexander_arntzen_yourleginnumber/Task5.txt"
########

# Vizualization and validation parameters
########
MODEL_LIST = np.arange(1)
SET_NAME = "final"
FOLDS = 10
#########

model_params = MODEL_PARAMS_CF
training_params = TRAINING_PARAMS_CF


def make_submission():
    # Data frame with data
    np.savetxt(PATH_SUBMISSION, x_opt_curve_, delimiter=" ")


def plot_task5(
    x_sampled, pred_sampled, x_minimizer, plot_name, path_figures=PATH_FIGURES
):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("D")
    ax.set_ylabel("v")
    pred_scatter = ax.scatter(x_sampled[:, 0], x_sampled[:, 1], c=pred_sampled[:, 0])
    ax.scatter(
        x_minimizer[:, 0],
        x_minimizer[:, 1],
        color="orange",
        label="minimizer points",
        s=5,
    )
    pred_legend = ax.legend(
        *pred_scatter.legend_elements(), title="cost", loc="upper right"
    )
    ax.add_artist(pred_legend)
    ax.legend()
    plt.savefig(f"{path_figures}/{plot_name}.pdf")


# functional style is great, but might not be optimized for python
def get_G(approx_CF, CF_ref=0.45):
    def G(Dv):
        # sum to make it work with simultaneously on many rows
        return torch.sum((CF_ref - approx_CF(Dv)) ** 2)

    return G


if __name__ == "__main__":
    # Data frame with data
    df_train = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32, sep=" ", header=None)
    df_train.columns = ["D", "v", "CF"]

    # Load data
    data_train_ = torch.tensor(df_train.values)
    x_train_ = data_train_[:, 0:2]
    y_train_ = data_train_[:, 2:3]

    # Normalize
    X_TRAIN_MAX = torch.tensor([20.0, 400.0])
    X_TRAIN_MIN = torch.tensor([2.0, 50.0])
    X_CENTER = X_TRAIN_MIN
    X_SCALE = X_TRAIN_MAX - X_TRAIN_MIN
    Y_CENTER = 0
    Y_SCALE = 1

    x_train = (x_train_ - X_CENTER) / X_SCALE
    y_train = (y_train_ - Y_CENTER) / Y_SCALE

    # structure data
    data_model_train = torch.utils.data.TensorDataset(x_train, y_train)

    model_params_iter = create_subdictionary_iterator(model_params)
    training_params_iter = create_subdictionary_iterator(training_params)

    # train model
    cv_results = k_fold_cv_grid(
        Model=FFNN,
        model_param_iter=model_params_iter,
        fit=fit_FFNN,
        training_param_iter=training_params_iter,
        data=data_model_train,
        init=init_xavier,
        partial=False,
        folds=FOLDS,
        verbose=True,
    )

    # create sobol starting points
    n_samples = 1000
    x_sampling_sobol = sobol.Sobol(dimension=2, graycode=True)
    x_sample = torch.tensor(x_sampling_sobol.gen_samples(n_samples).astype(np.float32))
    x_test_ = x_sample * X_SCALE + X_CENTER

    # plot models
    plot_kwargs1 = {
        "x_test": x_sample,
        "x_train": x_train,
        "y_train": y_train,
    }
    plot_result(
        path_figures=PATH_FIGURES,
        model_list=MODEL_LIST,
        plot_name=SET_NAME,
        **cv_results,
        plot_function=plot_model_scatter,
        function_kwargs=plot_kwargs1,
    )

    # optimization: find the point that optimize G
    # make the G function
    model_selected = cv_results["models"][0][6]
    CF_ref_ = 0.45
    CF_ref = (CF_ref_ - Y_CENTER) / Y_SCALE
    G = get_G(model_selected, CF_ref=CF_ref)

    # we optimize on the scaled problem because then constrains can be implemented
    x_opt_curve = x_sample.clone().detach()
    x_opt_curve = argmin(G, y_0=x_opt_curve, verbose=False, box_constraint=[0, 1])
    x_opt_curve_ = x_opt_curve * X_SCALE + X_CENTER

    g_train_ = (CF_ref_ - y_train_) ** 2
    g_test_ = ((CF_ref_ - model_selected(x_sample) * Y_SCALE - Y_CENTER) ** 2).detach()

    # plot optimization
    plot_task5(
        x_test_,
        g_test_,
        x_opt_curve_,
        plot_name=f"{SET_NAME}_optim_points",
        path_figures=PATH_FIGURES,
    )
    plot_task5(
        x_train_,
        g_train_,
        x_opt_curve_,
        plot_name=f"{SET_NAME}_optim_points_with_training_data",
        path_figures=PATH_FIGURES,
    )
