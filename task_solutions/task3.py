import pandas
import torch.utils.data
import numpy as np
import pandas as pd

from deepthermal.FFNN_model import fit_FFNN
from deepthermal.validation import (
    k_fold_cv_grid,
    create_subdictionary_iterator,
    get_scaled_results,
)
from deepthermal.plotting import plot_result, plot_result_sorted
from task_solutions.task3_model_params import (
    model_params,
    training_params,
    INPUT_WIDTH,
    LABEL_WIDTH,
    FOLDS,
    SET_NAME,
    DATA_COLUMN,
)
from deepthermal.forcasting import TimeSeriesDataset, get_structured_prediction, LSTM

# Path data
########
PATH_FIGURES = "figures/task3"
PATH_TRAINING_DATA = "Task3/TrainingData.txt"
PATH_TESTING_POINTS = "Task3/TestingData.txt"
PATH_SUBMISSION = "alexander_arntzen_yourleginnumber/Task3.txt"
########


model_params_iter = create_subdictionary_iterator(model_params)
training_params_iter = create_subdictionary_iterator(training_params)


def plot_task3(model, t_pred, data_train, t_train, sequence_stride=None, prediction_only=False, **kwargs):
    t_indices, y_pred = get_structured_prediction(model, data_train, sequence_stride=sequence_stride,
                                                  prediction_only=prediction_only)
    y_train = data_train.data
    x_pred = torch.cat((t_train, t_pred))[t_indices]
    x_train = t_train

    plot_result_sorted(
        x_pred=x_pred, y_pred=y_pred, x_train=x_train, y_train=y_train, **kwargs
    )


def make_submission(model):
    # Data frame with data
    df_sub = pd.read_csv(PATH_SUBMISSION, dtype=np.float32)
    # df_sub = pd.DataFrame()
    df_sub["t"] = t_pred_[:, 0]
    _, y_pred = get_structured_prediction(model, data, prediction_only=True)
    df_sub[DATA_COLUMN] = y_pred[:, 0]
    df_sub.to_csv(PATH_SUBMISSION, index=False)


if __name__ == "__main__":
    # Data frame with data
    df_train = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32)
    df_test = pd.read_csv(PATH_TESTING_POINTS, dtype=np.float32)

    # Load data
    y_train_ = torch.tensor(df_train[[DATA_COLUMN]].values)
    t_train_ = torch.tensor(df_train[["t"]].values)
    t_pred_ = torch.tensor(df_test[["t"]].values)

    # Normalize data
    Y_MAX = torch.max(y_train_, dim=0)[0]
    Y_MIN = torch.min(y_train_, dim=0)[0]
    Y_CENTER = Y_MIN
    Y_SCALE = Y_MAX - Y_MIN
    y_train = (y_train_ - Y_CENTER) / Y_SCALE

    # structure data
    data_ = TimeSeriesDataset(
        y_train_, input_width=INPUT_WIDTH, label_width=LABEL_WIDTH
    )
    data = TimeSeriesDataset(y_train, input_width=INPUT_WIDTH, label_width=LABEL_WIDTH)

    model_params_iter = create_subdictionary_iterator(model_params)
    training_params_iter = create_subdictionary_iterator(training_params)

    # do training with cross validation
    cv_results = k_fold_cv_grid(
        Model=LSTM,
        model_param_iter=model_params_iter,
        fit=fit_FFNN,
        training_param_iter=training_params_iter,
        data=data,
        init=None,
        partial=False,
        folds=FOLDS,
        verbose=True,
    )
    cv_results_scaled = get_scaled_results(
        cv_results,
        x_center=Y_CENTER,
        x_scale=Y_SCALE,
        y_center=Y_CENTER,
        y_scale=Y_SCALE,
    )
    plot_kwargs = {
        "t_pred": t_pred_,
        "data_train": data_,
        "t_train": t_train_,
        "x_axis": "t",
        "y_axis": "T",
        "prediction_only": True,
    }
    plot_result(
        path_figures=PATH_FIGURES,
        plot_name=SET_NAME,
        **cv_results_scaled,
        plot_function=plot_task3,
        function_kwargs=plot_kwargs,
    )

# functions to make
# plot_result(x_test=x_test, x_train=x_train,\
# y_train=y_train, path_figures=PATH_FIGURES, **cv_results)
