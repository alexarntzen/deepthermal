DATA_COLUMN = "ts0"
SET_NAME = f"final_{DATA_COLUMN}"
FOLDS = 10

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
    "regularization_param": [1e-5],
    "optimizer": ["ADAM"],
    "learning_rate": [0.005],
}
MODEL_PARAMS_tf0 = {
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
}
model_params = MODEL_PARAMS_ts0
training_params = TRAINING_PARAMS_ts0
