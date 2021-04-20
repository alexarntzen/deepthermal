import torch.utils.data
import numpy as np
import pandas as pd

from deepthermal.FFNN_model import get_trained_nn_model
from deepthermal.validation import create_subdictionary_iterator, plot_model_history, plot_model_1d

FINAL_MODEL_PARAMS = {
    "input_dimension": [1],
    "output_dimension": [2],
    "n_hidden_layers": [5],
    "neurons": [250],
    "activation": ["relu"],
    "init_weight_seed": [24]
}
FINAL_TRAINING_PARAMS = {
    "num_epochs": [4000],
    "batch_size": [265 // 4],
    "regularization_exp": [2],
    "regularization_param": [1e-4],
    "optimizer": ["ADAM"],
}

########
PATH_FIGURES = "../figures"
PATH_TRAINING_DATA = "../Task1/TrainingData.txt"
PATH_TESTING_POINTS = "../Task1/TestingData.txt"

#########

FINAL_MODEL_PARAM = next(create_subdictionary_iterator(FINAL_MODEL_PARAMS))
FINAL_TRAINING_PARAM = next(create_subdictionary_iterator(FINAL_TRAINING_PARAMS))


def make_submission(model_params, training_params, plot=True):
    # Data frame with data
    df_train = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32)
    df_test = pd.read_csv(PATH_TESTING_POINTS, dtype=np.float32)

    # Load data
    x_train_ = torch.tensor(df_train[["t"]].values)
    y_train = torch.tensor(df_train[["tf0", "ts0"]].values)
    x_test_ = torch.tensor(df_test[["t"]].values)

    # Normalize values
    X_TRAIN_MEAN = torch.mean(x_train_)
    X_TRAIN_STD = torch.std(x_train_)
    x_train = (x_train_ - X_TRAIN_MEAN) / X_TRAIN_STD
    x_test = (x_test_ - X_TRAIN_MEAN) / X_TRAIN_STD
    model1 = get_trained_nn_model(model_params, training_params, x_train, y_train)
    if plot:
        plot_model_history(model1, "final_model")
        plot_model_1d(model1, x_test, "result_final_model", x_train, y_train)
    y_pred = model1(x_test).detach()
    df_test["tf0"] = y_pred[:, 0]
    df_test["ts0"] = y_pred[:, 1]
    df_test.to_csv(f"../alexander_arntzen_yourleginumber/Task1.txt", index=False)
