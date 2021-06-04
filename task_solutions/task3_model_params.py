INPUT_WIDTH = 34
LABEL_WIDTH = 34
MODEL_PARAMS_cf = {
    "input_dimension": [1],
    "output_dimension": [1],
    "neurons": [40],
}
TRAINING_PARAMS_cf = {
    "num_epochs": [3000],
    "batch_size": [265],
    "regularization_exp": [2],
    "regularization_param": [1e-6],
    "optimizer": ["ADAM"],
    "learning_rate": [0.001],
}
