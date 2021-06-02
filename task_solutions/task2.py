import torch
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
from sklearn import linear_model
from deepthermal.FFNN_model import FFNN, init_xavier
from deepthermal.validation import k_fold_cv_grid, create_subdictionary_iterator
from deepthermal.plotting import plot_model_scatter, plot_result, plot_compare_scatter
from deepthermal.multilevel import fit_multilevel_FFNN, MultilevelFFNN, MultilevelDataset, get_init_multilevel, \
    get_multilevel_RRSE
from task_solutions.task2_model_params import MODEL_PARAMS_cf, TRAINING_PARAMS_cf

# Path data
########
PATH_FIGURES = "figures/task2"
PATH_TRAINING_DATA_101 = "Task2/TrainingData_101.txt"
PATH_TRAINING_DATA_401 = "Task2/TrainingData_401.txt"
PATH_TRAINING_DATA_1601 = "Task2/TrainingData_1601.txt"
PATH_TESTING_POINTS = "Task2/TestingData.txt"
PATH_SOBOL_POINTS = "Task2/samples_sobol.txt"
PATH_SUBMISSION = "alexander_arntzen_yourleginumber/Task2.txt"
########

# Vizualization and validation parameters
########
DATA_COLUMN = "cf"
MODEL_LIST = np.arange(1)
SET_NAME = f"ML_1_{DATA_COLUMN}"
FOLDS = 5
#########

model_params = MODEL_PARAMS_cf
training_params = TRAINING_PARAMS_cf

model_params_iter = create_subdictionary_iterator(model_params)
training_params_iter = create_subdictionary_iterator(training_params)


# def make_submission(model):
#     # Data frame with data
#     df_test = pd.read_csv(PATH_SUBMISSION, dtype=np.float32)
#     x_test = (x_test_ - X_TRAIN_MEAN) / X_TRAIN_STD
#     y_pred = model(x_test).detach()
#     df_test[DATA_COLUMN] = y_pred[:, 0]
#     df_test.to_csv(PATH_SUBMISSION, index=False)
def get_detrasformed(data, sigma, scale_coefs):
    g = (data / scale_coefs - 1) / sigma
    y = (g + 1) / 2
    return y


if __name__ == "__main__":
    # Data frame with data
    df_train_1601 = pd.read_csv(PATH_TRAINING_DATA_1601, dtype=np.float32, sep=" ", header=None)
    df_train_401 = pd.read_csv(PATH_TRAINING_DATA_401, dtype=np.float32, sep=" ", header=None)
    df_train_101 = pd.read_csv(PATH_TRAINING_DATA_101, dtype=np.float32, sep=" ", header=None)
    df_test = pd.read_csv(PATH_TESTING_POINTS, dtype=np.float32, sep=" ", header=None)
    df_sobol = pd.read_csv(PATH_SOBOL_POINTS, dtype=np.float32, sep=" ", header=None)

    # Load data
    x_train_101_ = torch.tensor(df_train_101.values)[:, :8]
    y_train_101_ = torch.tensor(df_train_101.values)[:, 8:9]
    x_train_401_ = torch.tensor(df_train_401.values)[:, :8]
    y_train_401_ = torch.tensor(df_train_401.values)[:, 8:9]
    x_train_1601_ = torch.tensor(df_train_1601.values)[:, :8]
    y_train_1601_ = torch.tensor(df_train_1601.values)[:, 8:9]
    x_test = torch.tensor(df_test.values)
    x_sobol = torch.tensor(df_sobol.values)

    # find the parametrization
    linreg_intercepts = torch.zeros(8)
    linreg_coefs = torch.zeros(8)
    for i in range(8):
        lin_reg = linear_model.LinearRegression().fit(2 * x_sobol[:, i:i + 1] - 1, x_train_101_[:, i:i + 1])
        linreg_coefs[i] = lin_reg.coef_.item()
        linreg_intercepts[i] = lin_reg.intercept_.item()
    SIGMA = torch.mean(linreg_coefs / linreg_intercepts).item()
    SCAlE_COEFS = linreg_coefs / SIGMA

    # check that the right parametrization was found
    assert torch.max(torch.abs(x_train_101_ - SCAlE_COEFS * (1 + SIGMA * (2 * x_sobol - 1)))).item() < 1e-3

    # detransform to sobol scale
    x_train_101 = get_detrasformed(x_train_101_, sigma=SIGMA, scale_coefs=SCAlE_COEFS)
    x_train_401 = get_detrasformed(x_train_401_, sigma=SIGMA, scale_coefs=SCAlE_COEFS)
    x_train_1601 = get_detrasformed(x_train_1601_, sigma=SIGMA, scale_coefs=SCAlE_COEFS)
    assert torch.max(x_train_1601 - x_train_101[:160]).item() < 1e-10  # check points in right order

    Y_MEAN = torch.mean(y_train_101_)
    Y_STD = torch.std(y_train_101_)

    y_train_101 = (y_train_101_ - Y_MEAN) / Y_STD
    y_train_401 = (y_train_401_ - Y_MEAN) / Y_STD
    y_train_1601 = (y_train_1601_ - Y_MEAN) / Y_STD

    data_train = TensorDataset(x_train_101, y_train_101)
    data_ml = MultilevelDataset(x_train_101, (y_train_101, y_train_401, y_train_1601))
    model_params_iter = create_subdictionary_iterator(model_params)
    training_params_iter = create_subdictionary_iterator(training_params)

    cv_results = k_fold_cv_grid(
        Model=MultilevelFFNN,
        model_param_iter=model_params_iter,
        fit=fit_multilevel_FFNN,
        training_param_iter=training_params_iter,
        data=data_ml,
        init=get_init_multilevel(init_xavier),
        partial=True,
        folds=FOLDS,
        verbose=True,
        get_error=get_multilevel_RRSE
    )
    plot_kwargs = {
        "x_test": x_test,
        "x_train": data_train[:][0],
        "y_train": data_train[:][1],
    }

    plot_result(path_figures=PATH_FIGURES, plot_name=SET_NAME + "_compare_", **cv_results,
                plot_function=plot_model_scatter,
                function_kwargs=plot_kwargs)
    plot_result(path_figures=PATH_FIGURES, plot_name=SET_NAME + "_vis", **cv_results,
                plot_function=plot_compare_scatter,
                function_kwargs=plot_kwargs, history=False)
