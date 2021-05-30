INPUT_WIDTH = 21
LABEL_WIDTH = 34
MODEL_PARAMS_cf = {
    "input_dimension": [INPUT_WIDTH],
    "output_dimension": [LABEL_WIDTH],
    "n_hidden_layers": [10],
    "neurons": [40],
    "activation": ["relu"],
}
TRAINING_PARAMS_cf = {
    "num_epochs": [3000],
    "batch_size": [265],
    "regularization_exp": [2],
    "regularization_param": [1e-6],
    "optimizer": ["ADAM"],
    "learning_rate": [0.001],
}
