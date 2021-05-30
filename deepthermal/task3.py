import torch.utils.data
import numpy as np
import pandas as pd

from deepthermal.FFNN_model import FFNN, fit_FFNN, init_xavier
from deepthermal.validation import k_fold_cv_grid, create_subdictionary_iterator
from deepthermal.plotting import plot_model_history, plot_result_sorted
from deepthermal.task3_model_params import (
    MODEL_PARAMS_cf,
    TRAINING_PARAMS_cf,
    INPUT_WIDTH,
    LABEL_WIDTH,
)
from deepthermal.forcasting import TimeSeriesDataset, get_structured_prediction

# Path data
########
PATH_FIGURES = "figures/task3"
PATH_TRAINING_DATA = "Task3/TrainingData.txt"
PATH_TESTING_POINTS = "Task3/TestingData.txt"
PATH_SUBMISSION = "alexander_arntzen_yourleginumber/Task3.txt"
########

# Vizualization and validation parameters
########
DATA_COLUMN = "ts0"
MODEL_LIST = np.arange(1)
SET_NAME = f"initial_{INPUT_WIDTH}_{DATA_COLUMN}"
FOLDS = 10
#########
# encoder decoder lstm

model_params = MODEL_PARAMS_cf
training_params = TRAINING_PARAMS_cf

model_params_iter = create_subdictionary_iterator(model_params)
training_params_iter = create_subdictionary_iterator(training_params)


# printing model errors
def print_model_errors(rel_val_errors, **kwargs):
    for i, rel_val_error_list in enumerate(rel_val_errors):
        avg_error = sum(rel_val_error_list) / len(rel_val_error_list)
        print(f"Model {i} validation error: {avg_error * 100}%")


def plot_result(
    models,
    data_train,
    loss_history_trains,
    loss_history_vals,
    rel_val_errors,
    t_train=None,
    t_pred=None,
    **kwargs,
):
    print_model_errors(rel_val_errors)
    for i in MODEL_LIST:
        plot_model_history(
            models[i],
            loss_history_trains[i],
            loss_history_vals[i],
            plot_name=(SET_NAME + f"_{i}"),
            path_figures=PATH_FIGURES,
        )
        for j in range(len(models[i])):
            t_indices, y_pred = get_structured_prediction(models[i][j], data_train)

            x_pred = torch.cat((t_train, t_pred))[t_indices]
            x_train = t_train
            y_train = data_train.data

            plot_result_sorted(
                x_pred=x_pred,
                y_pred=y_pred,
                x_train=x_train,
                y_train=y_train,
                plot_name=f"{SET_NAME}_{i}_{j}",
                path_figures=PATH_FIGURES,
            )

    # def make_submission(model):
    #     # Data frame with data
    #     df_test = pd.read_csv(PATH_SUBMISSION, dtype=np.float32)
    #     x_test = (x_test_ - X_TRAIN_MEAN) / X_TRAIN_STD
    #     y_pred = model(x_test).detach()
    #     df_test[DATA_COLUMN] = y_pred[:, 0]
    #     df_test.to_csv(PATH_SUBMISSION, index=False)


if __name__ == "__main__":
    # Data frame with data
    df_train = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32)
    df_test = pd.read_csv(PATH_TESTING_POINTS, dtype=np.float32)

    # Load data
    data_train_ = torch.tensor(df_train[[DATA_COLUMN]].values)
    t_train = torch.tensor(df_train[["t"]].values)
    t_pred = torch.tensor(df_test[["t"]].values)

    # Standardise values
    # Normalize values
    X_TRAIN_MEAN = torch.mean(data_train_)
    X_TRAIN_STD = torch.std(data_train_)
    data_train = (data_train_ - X_TRAIN_MEAN) / X_TRAIN_STD

    data = TimeSeriesDataset(
        data_train[:, 0], input_width=INPUT_WIDTH, label_width=LABEL_WIDTH
    )

    model_params_iter = create_subdictionary_iterator(model_params)
    training_params_iter = create_subdictionary_iterator(training_params)

    cv_results = k_fold_cv_grid(
        Model=FFNN,
        model_param_iter=model_params_iter,
        fit=fit_FFNN,
        training_param_iter=training_params_iter,
        data=data,
        init=init_xavier,
        partial=True,
        folds=FOLDS,
        verbose=True,
    )

    plot_result(
        data_train=data,
        t_train=t_train,
        t_pred=t_pred,
        path_figures=PATH_FIGURES,
        **cv_results,
    )

# functions to make
# plot_result(x_test=x_test, x_train=x_train, y_train=y_train, path_figures=PATH_FIGURES, **cv_results)
