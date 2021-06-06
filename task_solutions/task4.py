import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepthermal.FFNN_model import FFNN, fit_FFNN, init_xavier
from deepthermal.validation import (
    k_fold_cv_grid,
    get_scaled_results,
    create_subdictionary_iterator,
)
from deepthermal.plotting import plot_result
from task_solutions.task4_model_params import (
    MODEL_PARAMS_T,
    TRAINING_PARAMS_T,
    V_GUESS_,
    SET_NAME,
    FOLDS,
)
from deepthermal.optimization import argmin

# Path data
########
PATH_FIGURES = "figures/task4"
PATH_TRAINING_DATA = "Task4/TrainingData.txt"
PATH_MEASURED_DATA = "Task4/MeasuredData.txt"
PATH_SUBMISSION = "alexander_arntzen_yourleginnumber/Task4.txt"
########

model_params = MODEL_PARAMS_T
training_params = TRAINING_PARAMS_T


def plot_task4(
    data_train,
    data_measured,
    plot_name,
    model=None,
    path_figures=PATH_FIGURES,
    v_guess=0.5,
):
    num_t = data_measured.shape[0]
    data_plotting_guess = torch.zeros((num_t, 2))
    data_plotting_guess[:, 0] = data_measured[:, 0]
    data_plotting_guess[:, 1] = v_guess
    temp_pred = model(data_train[:, 0:2]).detach()
    check_v_pred = model(data_plotting_guess).detach()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("t")
    ax.set_ylabel("T")
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
        data_plotting_guess[:, 0],
        check_v_pred[:, 0],
        lw=2,
        color="blue",
        label=f"pred (u={v_guess})",
    )
    legend = ax.legend(loc="upper left")

    scatter = ax.scatter(
        data_train[:, 0],
        data_train[:, 2],
        c=data_train[:, 1],
        label=data_train[:, 1],
        marker=".",
    )
    ax.legend(*scatter.legend_elements(), loc="lower right", title="u")
    ax.add_artist(legend)
    fig.savefig(f"{path_figures}/{plot_name}.pdf")
    plt.close(fig)


# functional style is great, but might not be optimized for python
# big T is temperature, little t is time
def get_G(t, T, approx_T):
    def G(v):
        # creates a tensor with rows: [t_i, v]
        v_tensor = torch.tile(v, t.shape)
        data = torch.cat((t, v_tensor), dim=1)
        return torch.mean((T - approx_T(data)) ** 2)

    return G


if __name__ == "__main__":
    # Data frame with data
    df_train = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32, sep=" ", header=None)
    df_measured = pd.read_csv(
        PATH_MEASURED_DATA, dtype=np.float32, sep=" ", header=None
    )

    # Load data: 0 time, 1 velocity, 2 Temperature
    data_train_ = torch.tensor(df_train.values)
    x_train_ = data_train_[:, 0:2]
    y_train_ = data_train_[:, 2:3]

    # 0 time, 1 Temperature
    data_measured_ = torch.tensor(df_measured.values, requires_grad=False)

    # Normalize data
    X_MAX = torch.max(x_train_, dim=0)[0]
    X_MIN = torch.min(x_train_, dim=0)[0]
    X_CENTER = X_MIN
    X_SCALE = X_MAX - X_MIN
    Y_MAX = torch.max(y_train_, dim=0)[0]
    Y_MIN = torch.min(y_train_, dim=0)[0]
    Y_CENTER = Y_MIN
    Y_SCALE = Y_MAX - Y_MIN
    MEASURED_CENTER = torch.tensor([X_CENTER[0], Y_CENTER[0]])
    MEASURED_SCALE = torch.tensor([X_SCALE[0], Y_SCALE[0]])
    x_train = (x_train_ - X_CENTER) / X_SCALE
    y_train = (y_train_ - Y_CENTER) / Y_SCALE
    data_measured = (data_measured_ - MEASURED_CENTER) / MEASURED_SCALE
    v_guess = (V_GUESS_ - X_CENTER[1]) / X_SCALE[1]

    # structure data
    data_train = torch.utils.data.TensorDataset(x_train, y_train)
    model_params_iter = create_subdictionary_iterator(model_params)
    training_params_iter = create_subdictionary_iterator(training_params)

    # train model
    cv_results = k_fold_cv_grid(
        Model=FFNN,
        model_param_iter=model_params_iter,
        fit=fit_FFNN,
        training_param_iter=training_params_iter,
        data=data_train,
        init=init_xavier,
        partial=False,
        folds=FOLDS,
        verbose=True,
    )
    cv_results_scaled = get_scaled_results(
        cv_results,
        x_center=X_CENTER,
        x_scale=X_SCALE,
        y_center=Y_CENTER,
        y_scale=Y_SCALE,
    )

    # chose models that fits best
    model = cv_results["models"][0][3]

    # optimization on the normalized problem
    t_tensor = data_measured[:, 0:1]
    T_tensor = data_measured[:, 1:2]
    G = get_G(t_tensor, T_tensor, model)
    v_opt = argmin(G, v_guess)
    v_opt_ = v_opt * X_SCALE[1] + X_CENTER[1]

    # # plot loss
    # v = torch.linspace(0.2, 1, 100)
    # g = torch.zeros_like(v)
    # for i in range(len(v)):
    #     g[i] = G(v[i])
    # plt.plot(v, g.detach())
    # plt.show()

    print("final_v: ", v_opt_)
    plot_kwargs2 = {
        "v_guess": v_opt_,
        "data_train": data_train_,
        "data_measured": data_measured_,
    }
    plot_result(
        path_figures=PATH_FIGURES,
        plot_name=SET_NAME,
        **cv_results_scaled,
        plot_function=plot_task4,
        function_kwargs=plot_kwargs2,
    )
