from deepthermal.FFNN_model import FFNN, init_xavier

DATA_COLUMN = "ts0"
SET_NAME = f"final_{DATA_COLUMN}"
FOLDS = 10

MODEL_PARAMS_ts0 = {
    "model": [FFNN],
    "input_dimension": [1],
    "output_dimension": [1],
    "n_hidden_layers": [10],
    "neurons": [40],
    "activation": ["relu"],
}
TRAINING_PARAMS_ts0 = {
    "num_epochs": [10000],
    "batch_size": [265],
    "regularization_exp": [2],
    "regularization_param": [1e-5],
    "optimizer": ["ADAM"],
    "learning_rate": [0.005],
    "init": [init_xavier],
}

MODEL_PARAMS_tf0 = {
    "model": [FFNN],
    "input_dimension": [1],
    "output_dimension": [1],
    "n_hidden_layers": [10],
    "neurons": [30],
    "activation": ["relu"],
}
TRAINING_PARAMS_tf0 = {
    "num_epochs": [10000],
    "batch_size": [265],
    "regularization_exp": [2],
    "regularization_param": [1e-3],
    "optimizer": ["ADAM"],
    "learning_rate": [0.005],
    "init": [init_xavier],
}
model_params = MODEL_PARAMS_ts0
training_params = TRAINING_PARAMS_ts0
