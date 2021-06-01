MODEL_PARAMS_CF = {
    "input_dimension": [2],
    "output_dimension": [1],
    "n_hidden_layers": [6],
    "neurons": [20],
    "activation": ["relu"],
}
TRAINING_PARAMS_CF = {
    "num_epochs": [1000],
    "batch_size": [128],
    "regularization_exp": [2],
    "regularization_param": [1e-6],
    "optimizer": ["ADAM"],
    "learning_rate": [0.0005],
}
