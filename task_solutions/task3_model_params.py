########
DATA_COLUMN = "ts0"
SET_NAME = f"final_{DATA_COLUMN}"

FOLDS = 5
#########

INPUT_WIDTH = 37
LABEL_WIDTH = 34
MODEL_PARAMS_tf0 = {
    "input_dimension": [1],
    "output_dimension": [1],
    "neurons": [25],
    "label_width": [LABEL_WIDTH],
}

TRAINING_PARAMS_tf0 = {
    "num_epochs": [5000],
    "batch_size": [265],
    "regularization_exp": [2],
    "regularization_param": [1e-6],
    "optimizer": ["LBFGS"],
    "learning_rate": [0.01],
}
MODEL_PARAMS_ts0 = {
    "input_dimension": [1],
    "output_dimension": [1],
    "neurons": [20],
    "label_width": [LABEL_WIDTH],
}

TRAINING_PARAMS_ts0 = {
    "num_epochs": [5000],
    "batch_size": [265],
    "regularization_exp": [2],
    "regularization_param": [1e-7],
    "optimizer": ["LBFGS"],
    "learning_rate": [0.01],
}
model_params = MODEL_PARAMS_ts0
training_params = TRAINING_PARAMS_ts0
