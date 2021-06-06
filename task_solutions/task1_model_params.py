DATA_COLUMN = "tf0"
SET_NAME = f"nosmooth_tanh_{DATA_COLUMN}"
FOLDS = 5

MODEL_PARAMS_ts0 = {
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
    "regularization_param": [1e-6],
    "optimizer": ["ADAM"],
    "learning_rate": [0.002],
}
MODEL_PARAMS_tf0 = {
    "input_dimension": [1],
    "output_dimension": [1],
    "n_hidden_layers": [7],
    "neurons": [30],
    "activation": ["tanh"],
}
TRAINING_PARAMS_tf0 = {
    "num_epochs": [5000],
    "batch_size": [265],
    "regularization_exp": [2],
    "regularization_param": [1e-3],
    "optimizer": ["ADAM"],
    "learning_rate": [0.002],
}

model_params = MODEL_PARAMS_tf0
training_params = TRAINING_PARAMS_tf0
