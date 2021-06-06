SET_NAME = "final_collected_100_"
FOLDS = 10
MODEL_PARAMS_cf = {
    "input_dimension": [8],
    "output_dimension": [1],
    "n_hidden_layers": [7],
    "neurons": [25],
    "activation": ["relu"],
}
TRAINING_PARAMS_cf = {
    "num_epochs": [7000],
    "batch_size": [265],
    "regularization_exp": [2],
    "regularization_param": [1e-4],
    "optimizer": ["ADAM"],
    "learning_rate": [0.005],
}
