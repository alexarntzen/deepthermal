import torch
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd

from deepthermal.FFNN_model import FFNN, fit_FFNN, init_xavier
from deepthermal.validation import (
    k_fold_cv_grid,
    create_subdictionary_iterator,
    get_scaled_results,
)
from deepthermal.plotting import plot_result, plot_model_1d
from task_solutions.task1_model_params import (
    model_params,
    training_params,
    DATA_COLUMN,
    SET_NAME,
    FOLDS,
)

########
PATH_FIGURES = "figures/task1"
PATH_TRAINING_DATA = "Task1/TrainingData.txt"
PATH_TESTING_POINTS = "Task1/TestingData.txt"
PATH_SUBMISSION = "alexander_arntzen_yourleginnumber/Task1.txt"
########


def make_submission(model):
    # Data frame with data
    df_sub = pd.read_csv(PATH_SUBMISSION, dtype=np.float32)
    # df_sub = pd.DataFrame()

    # this should be the scaled model!
    y_pred = model(x_test_).detach()
    df_sub[DATA_COLUMN] = y_pred[:, 0]
    df_sub.to_csv(PATH_SUBMISSION, index=False)


if __name__ == "__main__":
    # Data frame with data
    df_train = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32)
    df_test = pd.read_csv(PATH_TESTING_POINTS, dtype=np.float32)

    # Load data
    x_train_ = torch.tensor(df_train[["t"]].values)
    y_train_ = torch.tensor(df_train[[DATA_COLUMN]].values)
    x_test_ = torch.tensor(df_test[["t"]].values)

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

    data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_train, y_train)
    model_params_iter = create_subdictionary_iterator(model_params)
    training_params_iter = create_subdictionary_iterator(training_params)

    cv_results = k_fold_cv_grid(
        Model=FFNN,
        model_param_iter=model_params_iter,
        fit=fit_FFNN,
        training_param_iter=training_params_iter,
        data=data,
        val_data=val_data,
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
    plot_kwargs = {
        "x_test": x_test_,
        "x_train": x_train_,
        "y_train": y_train_,
        "x_axis": "t",
        "y_axis": "T",
    }

    plot_result(
        path_figures=PATH_FIGURES,
        plot_name=SET_NAME,
        **cv_results_scaled,
        plot_function=plot_model_1d,
        function_kwargs=plot_kwargs,
    )
