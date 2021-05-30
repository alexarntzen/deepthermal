import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepthermal.FFNN_model import FFNN, fit_FFNN, init_xavier
from deepthermal.validation import (
    k_fold_cv_grid,
    create_subdictionary_iterator,
)
from deepthermal.plotting import plot_result
from deepthermal.task4_model_params import MODEL_PARAMS_T, TRAINING_PARAMS_T, V_GUESS
from deepthermal.optimization import argmin

# some notes, use lbfgs?

# Path data
########
PATH_FIGURES = "figures/task4"
PATH_TRAINING_DATA = "Task4/TrainingData.txt"
PATH_MEASURED_DATA = "Task4/MeasuredData.txt"
PATH_SUBMISSION = "alexander_arntzen_yourleginumber/Task4.txt"
########

# Vizualization and validation parameters
########
MODEL_LIST = np.arange(1)
SET_NAME = "model_check_1"
FOLDS = 5
#########

model_params = MODEL_PARAMS_T
training_params = TRAINING_PARAMS_T


# def make_submission(model):
#     # Data frame with data
#     df_test = pd.read_csv(PATH_SUBMISSION, dtype=np.float32)
#     x_test = (x_test_ - X_TRAIN_MEAN) / X_TRAIN_STD
#     y_pred = model(x_test).detach()
#     df_test[DATA_COLUMN] = y_pred[:, 0]
#     df_test.to_csv(PATH_SUBMISSION, index=False)


def plot_task4(
    data_train,
    data_measured,
    plot_name,
    model=None,
    path_figures=PATH_FIGURES,
    v_guess=0.5,
):
    num_t = data_measured.shape[0]
    data_plotting = torch.zeros((num_t, 2))
    data_plotting[:, 0] = data_measured[:, 0]
    data_plotting[:, 1] = v_guess
    temp_pred = model(data_train[:, 0:2]).detach()
    check_v_pred = model(data_plotting).detach()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("t")
    ax.set_ylabel("T")
    ax.set_title("Compare mesurements to computed solutions")
    ax.plot(
        data_measured[:, 0],
        data_measured[:, 1],
        lw=1,
        color="r",
        alpha=0.5,
        label="observations",
    )
    plt.scatter(
        data_train[:, 0],
        temp_pred,
        color="orange",
        label="model prediction",
        marker="x",
    )
    ax.plot(
        data_plotting[:, 0],
        check_v_pred[:, 0],
        lw=2,
        color="blue",
        label=f"pred (v={v_guess})",
    )
    legend = ax.legend(loc="upper left")

    scatter = ax.scatter(
        data_train[:, 0],
        data_train[:, 2],
        c=data_train[:, 1],
        label=data_train[:, 1],
        marker=".",
    )
    ax.legend(*scatter.legend_elements(), loc="lower right", title="v")
    ax.add_artist(legend)
    plt.savefig(f"{path_figures}/{plot_name}.pdf")


# functional style is great, but might not be optimized for python
def get_G(t, T, approx_T):
    def G(v):
        data = torch.stack((t, v.tile((t.shape[0],))), dim=1)
        return torch.mean((T - approx_T(data)) ** 2)

    return G


if __name__ == "__main__":
    # Data frame with data
    df_train = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32, sep=" ", header=None)
    df_measured = pd.read_csv(
        PATH_MEASURED_DATA, dtype=np.float32, sep=" ", header=None
    )
    df_train.columns = ["t", "v", "T"]
    df_measured.columns = ["t", "T"]

    # Load data
    data_train_ = torch.tensor(
        df_train.sort_values(by=["v"]).values
    )  # 0 time, 1 Temperature
    data_measured_ = torch.tensor(
        df_measured.values, requires_grad=False
    )  # 0 time, 1 velocity, 2 Temperature

    # Normalize
    DATA_TRAIN_MAX = torch.max(data_train_, dim=0)[0]
    DATA_TRAIN_MIN = torch.min(data_train_, dim=0)[0]
    DATA_TRAIN_MIN[0] = 0
    DATA_TRAIN_MAX[0] = 0.25
    DATA_CENTER = DATA_TRAIN_MIN
    DATA_SCALE = DATA_TRAIN_MAX - DATA_TRAIN_MIN

    data_train = (data_train_ - DATA_CENTER) / DATA_SCALE
    data_measured = (data_measured_ - DATA_CENTER[[0, 2]]) / DATA_SCALE[[0, 2]]
    v_guess_ = (V_GUESS - DATA_CENTER[1]) / DATA_SCALE[1]

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

    # plot model
    SET_NAME_1 = SET_NAME + "1"
    plot_kwargs1 = {
        "v_guess": v_guess_,
        "data_train": data_train,
        "data_measured": data_measured,
    }
    plot_result(
        path_figures=PATH_FIGURES,
        model_list=MODEL_LIST,
        plot_name=SET_NAME_1,
        **cv_results,
        plot_function=plot_task4,
        function_kwargs=plot_kwargs1,
    )

    # optimization
    t_tensor = data_measured[:, 0]
    T_tensor = data_measured[:, 1]
    model = cv_results["models"][0][0]
    G = get_G(t_tensor, T_tensor, model)
    v_opt = argmin(G, v_guess_)
    print("final_v: ", v_opt, v_guess_)

    SET_NAME_2 = SET_NAME + "2"
    plot_kwargs2 = {
        "v_guess": v_opt,
        "data_train": data_train,
        "data_measured": data_measured,
    }
    plot_result(
        path_figures=PATH_FIGURES,
        model_list=MODEL_LIST,
        plot_name=SET_NAME_2,
        **cv_results,
        plot_function=plot_task4,
        function_kwargs=plot_kwargs2,
    )
