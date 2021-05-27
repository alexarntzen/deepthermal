import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepthermal.FFNN_model import FFNN, fit_FFNN, init_xavier
from deepthermal.validation import k_fold_cv_grid, create_subdictionary_iterator, print_model_errors
from deepthermal.plotting import get_disc_str, plot_model_history, plot_result_sorted
from deepthermal.task4_model_params import MODEL_PARAMS_T, TRAINING_PARAMS_T
from deepthermal.forcasting import TimeSeriesDataset, get_structured_prediction

# some notes, use lbfgs?

# Path data
########
PATH_FIGURES = "figures/task4"
PATH_TRAINING_DATA = "Task4/TrainingData.txt"
PATH_MEASURED_DATA = "Task4/MeasuredData.txt"
PATH_SUBMISSION = "alexander_arntzen_yourleginumber/Task4.txt"
########

# Vizualization and validation parameterso
########
MODEL_LIST = np.arange(1)
SET_NAME = f"initial_1"
FOLDS = 5
#########

model_params = MODEL_PARAMS_T
training_params = TRAINING_PARAMS_T


def plot_result(data_train, data_measured, models, loss_history_trains, loss_history_vals, rel_val_errors,
                v_guess=0.5, **kwargs):
    print_model_errors(rel_val_errors)
    for i in MODEL_LIST:
        plot_model_history(models[i], loss_history_trains[i], loss_history_vals[i], plot_name=(SET_NAME + f"_{i}"),
                           path_figures=PATH_FIGURES)
        for j in range(len(models[i])):
            plot_task4(data_measured, data_train, plot_name=f"{SET_NAME}_{i}_{j}", model=models[i][j],
                       path_figures=PATH_FIGURES, v_guess=v_guess)


# def make_submission(model):
#     # Data frame with data
#     df_test = pd.read_csv(PATH_SUBMISSION, dtype=np.float32)
#     x_test = (x_test_ - X_TRAIN_MEAN) / X_TRAIN_STD
#     y_pred = model(x_test).detach()
#     df_test[DATA_COLUMN] = y_pred[:, 0]
#     df_test.to_csv(PATH_SUBMISSION, index=False)


def plot_task4(data_measured, data_train, plot_name, model=None, path_figures=PATH_FIGURES, v_guess=0.5):
    num_t = data_measured.shape[0]
    data_plotting = torch.zeros((num_t, 2))
    data_plotting[:, 0] = data_measured[:, 0]
    data_plotting[:, 1] = v_guess
    temp_pred = model(data_train[:, 0:2]).detach()

    fig, ax = plt.subplots()
    ax.set_xlabel("t")
    ax.set_ylabel("T")
    ax.set_title("Compare mesurements to computed solutions")
    # ax.plot(data_measured[:, 0], data_measured[:, 1], lw=1, color="r", alpha=0.5, label="observations")
    plt.scatter(data_train[:, 0], temp_pred, color="orange", label="model prediction", marker="x")
    legend = ax.legend()
    scatter = ax.scatter(data_train[:, 0], data_train[:, 2], c=data_train[:, 1], label=data_train[:, 1], marker=".")
    ax.legend(*scatter.legend_elements(),
              loc="lower right", title="v")
    ax.add_artist(legend)
    plt.savefig(f"{path_figures}/{plot_name}.pdf")


if __name__ == "__main__":
    # Data frame with data
    df_train = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32, sep=" ", header=None)
    df_measured = pd.read_csv(PATH_MEASURED_DATA, dtype=np.float32, sep=" ", header=None)
    df_train.columns = ['t', 'v', 'T']
    df_measured.columns = ['t', 'T']

    # Load data
    data_train_ = torch.tensor(df_train.sort_values(by=["v"]).values)  # 0 time, 1 Temperature
    data_measured_ = torch.tensor(df_measured.values)  # 0 time, 1 velocity, 2 Temperature

    # Normalize
    DATA_TRAIN_MAX = torch.max(data_train_, dim=0)[0]
    DATA_TRAIN_MIN = torch.min(data_train_, dim=0)[0]
    # DATA_T_MEAN = torch.mean(data_train_[:, 2])
    # DATA_T_STD = torch.std(data_train_[:, 2])
    DATA_TRAIN_MIN[0] = 0
    DATA_TRAIN_MAX[0] = 0.25
    DATA_CENTER = DATA_TRAIN_MIN
    DATA_SCALE = DATA_TRAIN_MAX - DATA_TRAIN_MIN
    # DATA_CENTER[2] = DATA_T_MEAN
    # DATA_SCALE[2] = DATA_T_STD

    # data_train_ = data_train_[128*3:128*4]
    data_train = (data_train_ - DATA_CENTER) / DATA_SCALE
    data_measured = (data_measured_ - DATA_CENTER[[0, 2]]) / DATA_SCALE[[0, 2]]

    # structure data
    data_model_train = torch.utils.data.TensorDataset(data_train[:, 0:2], data_train[:, 2:3])
    model_params_iter = create_subdictionary_iterator(model_params)
    training_params_iter = create_subdictionary_iterator(training_params)

    #
    cv_results = k_fold_cv_grid(Model=FFNN,
                                model_param_iter=model_params_iter,
                                fit=fit_FFNN,
                                training_param_iter=training_params_iter,
                                data=data_model_train,
                                init=init_xavier,
                                partial=True,
                                folds=FOLDS,
                                verbose=True)

    # functions to make
    plot_result(data_train, data_measured, path_figures=PATH_FIGURES, **cv_results, v_guess=0.5)
