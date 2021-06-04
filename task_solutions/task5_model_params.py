MODEL_PARAMS_CF = {
    "input_dimension": [2],
    "output_dimension": [1],
    "n_hidden_layers": [3],
    "neurons": [30],
    "activation": ["tanh"],
}
TRAINING_PARAMS_CF = {
    "num_epochs": [3000],
    "batch_size": [50],
    "regularization_exp": [2],
    "regularization_param": [1e-5],
    "optimizer": ["LBFGS"],
    "learning_rate": [0.01],
}
