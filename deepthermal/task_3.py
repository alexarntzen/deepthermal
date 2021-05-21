import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepthermal.FFNN_model import FFNN, fit_FFNN, init_xavier
from deepthermal.validation import k_fold_cv_grid, create_subdictionary_iterator
from deepthermal.plotting import get_disc_str, plot_model_history, plot_model_scatter
from deepthermal.task3_model_params import MODEL_PARAMS_cf, TRAINING_PARAMS_cf

# Path data
########
PATH_FIGURES = "figures/task3"
PATH_TRAINING_DATA = "Task3/TrainingData.txt"
PATH_TESTING_POINTS = "Task3/TestingData.txt"
PATH_SUBMISSION = "alexander_arntzen_yourleginumber/Task3.txt"
########

# Vizualization and validation parameters
########
DATA_COLUMN = "cf"
MODEL_LIST = np.arange(1)
SET_NAME = f"initial_1_{DATA_COLUMN}"
FOLDS = 5
#########

model_params = MODEL_PARAMS_cf
training_params = TRAINING_PARAMS_cf

model_params_iter = create_subdictionary_iterator(model_params)
training_params_iter = create_subdictionary_iterator(training_params)


## printing model errors
def print_model_errors(rel_val_errors, **kwargs):
    for i, rel_val_error_list in enumerate(rel_val_errors):
        avg_error = sum(rel_val_error_list) / len(rel_val_error_list)
        print(f"Model {i} validation error: {avg_error * 100}%")


def plot_result(models, loss_history_trains, loss_history_vals, rel_val_errors, x_test, x_train, y_train, **kwargs):
    print_model_errors(rel_val_errors)
    for i in MODEL_LIST:
        plot_model_history(models[i], loss_history_trains[i], loss_history_vals[i], plot_name=(SET_NAME + f"_{i}"),
                           path_figures=PATH_FIGURES)
        for j in range(len(models[i])):
            plot_model_scatter(models[i][j], x_test, f"{SET_NAME}_{i}_{j}", x_train, y_train,
                               path_figures=PATH_FIGURES)


# def make_submission(model):
#     # Data frame with data
#     df_test = pd.read_csv(PATH_SUBMISSION, dtype=np.float32)
#     x_test = (x_test_ - X_TRAIN_MEAN) / X_TRAIN_STD
#     y_pred = model(x_test).detach()
#     df_test[DATA_COLUMN] = y_pred[:, 0]
#     df_test.to_csv(PATH_SUBMISSION, index=False)


if __name__ == "__main__":
    # Data frame with data
    df_train = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32, sep=" ", header=None)
    df_test = pd.read_csv(PATH_TESTING_POINTS, dtype=np.float32, sep=" ", header=None)

    # Load data
    x_train_ = torch.tensor(df_train.values)[:, :8]
    y_train_ = torch.tensor(df_train.values)[:, 8:9]
    x_test_ = torch.tensor(df_test.values)

    # Standardise values
    X_TRAIN_MEAN = torch.tensor(df_train.mean())[:8]
    X_TRAIN_STD = torch.tensor(df_train.std())[:8]
    Y_TRAIN_MEAN = torch.tensor(df_train.mean())[8:9]
    Y_TRAIN_STD = torch.tensor(df_train.std())[8:9]

    X_0 = X_TRAIN_MEAN
    SIGMA = torch.mean(torch.abs(x_train_ / X_0 - 1))*2

    # Transformed to G since, G has standard mean and std
    x_train = (x_train_ / X_0 -1) / SIGMA
    # Scale G(y) = 2y -1
    x_test = 2 * x_test_ - 1
    y_train = (y_train_ - Y_TRAIN_MEAN) / Y_TRAIN_STD

    # Number of training samples
    n_samples = len(df_train.index)

    model_params_iter = create_subdictionary_iterator(model_params)
    training_params_iter = create_subdictionary_iterator(training_params)

    cv_results = k_fold_cv_grid(Model=FFNN,
                                model_param_iter=model_params_iter,
                                fit=fit_FFNN,
                                training_param_iter=training_params_iter,
                                x=x_train,
                                y=y_train,
                                init=init_xavier,
                                partial=True,
                                folds=FOLDS,
                                verbose=True)

    plot_result(x_test=x_test, x_train=x_train, y_train=y_train, path_figures=PATH_FIGURES, **cv_results)
# functions to make
# plot_result(x_test=x_test, x_train=x_train, y_train=y_train, path_figures=PATH_FIGURES, **cv_results)
