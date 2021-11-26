from deepthermal.FFNN_model import FFNN, init_xavier

SET_NAME = "final"
FOLDS = 5

MODEL_PARAMS_T = {
    "model": [FFNN],
    "input_dimension": [2],
    "output_dimension": [1],
    "n_hidden_layers": [9],
    "neurons": [25],
    "activation": ["relu"],
}
TRAINING_PARAMS_T = {
    "num_epochs": [3000],
    "batch_size": [128],
    "regularization_exp": [2],
    "regularization_param": [1e-5],
    "optimizer": ["ADAM"],
    "learning_rate": [0.001],
    "init": [init_xavier],
}

V_GUESS_ = 15.24
