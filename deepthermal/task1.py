import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deepthermal.FFNN_model import FFNN, fit_FFNN, init_xavier
from deepthermal.validation import k_fold_CV_grid, create_subdictionary_iterator, get_disc_str, plot_model_history
from deepthermal.task1_model_params import MODEL_PARAMS_tf0, TRAINING_PARAMS_tf0, MODEL_PARAMS_ts0, TRAINING_PARAMS_ts0

########
PATH_FIGURES = "../figures/task1"
PATH_TRAINING_DATA = "../Task1/TrainingData.txt"
PATH_TESTING_POINTS = "../Task1/TestingData.txt"
PATH_SUBMISSION = "../alexander_arntzen_yourleginumber/Task1.txt"
########

DATA_COLUMN = "ts0"
MODEL_LIST = np.arange(1)
SET_NAME = f"relu_10_{DATA_COLUMN}"

model_params = MODEL_PARAMS_ts0
training_params = TRAINING_PARAMS_ts0

# Data frame with data
df_train = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32)
df_test = pd.read_csv(PATH_TESTING_POINTS, dtype=np.float32)

# Load data
x_train_ = torch.tensor(df_train[["t"]].values)
y_train = torch.tensor(df_train[[DATA_COLUMN]].values)
x_test_ = torch.tensor(df_test[["t"]].values)

# Normalize values
X_TRAIN_MEAN = torch.mean(x_train_)
X_TRAIN_STD = torch.std(x_train_)
x_train = (x_train_ - X_TRAIN_MEAN) / X_TRAIN_STD
x_test = (x_test_ - X_TRAIN_MEAN) / X_TRAIN_STD

# Number of training samples
n_samples = len(df_train.index)

model_params_iter = create_subdictionary_iterator(model_params)
training_params_iter = create_subdictionary_iterator(training_params)

models, rel_training_error, rel_val_errors = k_fold_CV_grid(Model=FFNN, model_param_iter=model_params_iter,
                                                            fit=fit_FFNN, training_param_iter=training_params_iter,
                                                            x=x_train, y=y_train, init=init_xavier, partial=False,
                                                            verbose=True)


## printing model errors
def print_model_errors(rel_val_errors):
    for i, rel_val_error_list in enumerate(rel_val_errors):
        avg_error = sum(rel_val_error_list) / len(rel_val_error_list)
        print(f"Model {i} validation error: {avg_error * 100}%")


# plot visualization
def plot_models(model_number_list, models=models, plot_name="vis_model", ):
    k = len(models[0])
    # num_models = len(model_number_list)
    for model_number in model_number_list:
        fig, axs = plt.subplots(1, k, figsize=(8 * k, 6))
        fig.suptitle(f"Model: {get_disc_str(models[model_number][0])}")

        for k_, ax in enumerate(fig.axes):
            ax.scatter(x_train[:, 0], y_train[:, 0], label=f"{DATA_COLUMN}_train")

            ax.plot(x_test, models[model_number][k_](x_test)[:, 0].detach(), label=f"{DATA_COLUMN}_pred", lw=2, ls="-.",
                    color="black", )

            ax.set_xlabel("t")
            ax.set_ylabel("T")
            ax.legend()

        fig.savefig(f"{PATH_FIGURES}/{plot_name}_{model_number}.pdf")
        plt.close(fig)


def plot_result():
    print_model_errors(rel_val_errors)
    plot_models(MODEL_LIST, models, "result_" + SET_NAME)
    for i in MODEL_LIST:
        plot_model_history(model=models[i][0], model_name=(SET_NAME + f"_{i}"), path_figures=PATH_FIGURES)


def make_submission(model):
    # Data frame with data
    df_test = pd.read_csv(PATH_SUBMISSION, dtype=np.float32)
    x_test = (x_test_ - X_TRAIN_MEAN) / X_TRAIN_STD
    y_pred = model(x_test).detach()
    df_test[DATA_COLUMN] = y_pred[:, 0]
    df_test.to_csv(PATH_SUBMISSION, index=False)


# plot_model_1d(model=, x_test, "result_final_model_tf0", x_train, y_train)
