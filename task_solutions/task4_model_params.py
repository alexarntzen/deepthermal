SET_NAME = "final_1"
FOLDS = 5

MODEL_PARAMS_T = {
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
}

V_GUESS_ = 15.24
