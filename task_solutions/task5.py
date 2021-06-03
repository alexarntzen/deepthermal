# might be linear
# takes about an hour for 1000 points
# cannot use ismo because you cannot generate data with training data
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
PATH_SUBMISSION = "alexander_arntzen_yourleginumber/Task5.txt"
########

# Vizualization and validation parameters
########
MODEL_LIST = np.arange(1)
SET_NAME = "model_check_few_points_0"
FOLDS = 10
#########

model_params = MODEL_PARAMS_CF
training_params = TRAINING_PARAMS_CF


# def make_submission(model):
#     # Data frame with data
#     df_test = pd.read_csv(PATH_SUBMISSION, dtype=np.float32)
#     x_test = (x_test_ - X_TRAIN_MEAN) / X_TRAIN_STD
#     y_pred = model(x_test).detach()
#     df_test[DATA_COLUMN] = y_pred[:, 0]
#     df_test.to_csv(PATH_SUBMISSION, index=False)


def plot_task5(
    x_sampled, pred_sampled, x_minimizer, plot_name, path_figures=PATH_FIGURES
):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("D")
    ax.set_ylabel("v")
    ax.set_title("Cost function minimizer vizualization")
    pred_scatter = ax.scatter(x_sampled[:, 0], x_sampled[:, 1], c=pred_sampled[:, 0])
    ax.scatter(
        x_minimizer[:, 0], x_minimizer[:, 1], color="orange", label="minimizer points"
    )
    pred_legend = ax.legend(*pred_scatter.legend_elements(), title="sampled points")
    ax.add_artist(pred_legend)
    ax.legend()
    plt.savefig(f"{path_figures}/{plot_name}.pdf")


# functional style is great, but might not be optimized for python
def get_G(approx_CF, CF_ref=0.45):
    def G(Dv):
        return (CF_ref - approx_CF(Dv)) ** 2

    return G


if __name__ == "__main__":
    # Data frame with data
    df_train = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32, sep=" ", header=None)
    df_train.columns = ["D", "v", "CF"]

    # Load data
    data_train_ = torch.tensor(df_train.values)

    # Normalize and standardize
    DATA_TRAIN_MAX = torch.tensor([50.0, 400.0, 1.0])
    DATA_TRAIN_MIN = torch.tensor([2.0, 50.0, 0.0])
    DATA_CENTER = DATA_TRAIN_MIN
    DATA_SCALE = DATA_TRAIN_MAX - DATA_TRAIN_MIN
    DATA_CENTER[2] = torch.mean(data_train_[:, 2])
    DATA_SCALE[2] = torch.std(data_train_[:, 2])
    data_train = (data_train_ - DATA_CENTER) / DATA_SCALE

    # structure data
    data_model_train = torch.utils.data.TensorDataset(
        data_train[:, 0:2], data_train[:, 2:3]
    )
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
        partial=True,
        folds=FOLDS,
        verbose=True,
    )

    # plot model tp check that it is right
    n_samples = 20

    x_sampling_sobol = sobol.Sobol(dimension=2, graycode=True)
    x_sample = x_sampling_sobol.gen_samples(n_samples).astype(np.float32)
    x_test = torch.tensor(
        x_sample
    )  # *DATA_SCALE[0:2] +DATA_CENTER to get physical data
    plot_kwargs1 = {
        "x_test": x_test,
        "x_train": data_train[:, 0:2],
        "y_train": data_train[:, 2:3],
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
    x_opt_curve = x_test.clone().detach()
    G = get_G(cv_results["models"][0][0])
    for i in range(n_samples):
        x_opt_curve[i] = argmin(
            G, y_0=x_opt_curve[i], verbose=False, box_constraint=[0, 1]
        )
        print(i)
    g_test = G(x_test).detach()
    plot_task5(
        x_test,
        g_test,
        x_opt_curve,
        plot_name=f"{SET_NAME}_optim_points",
        path_figures=PATH_FIGURES,
    )
