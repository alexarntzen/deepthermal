MODEL_PARAMS_cf = {
    "input_dimension": [8],
    "output_dimension": [1],
    "n_hidden_layers": [7],
    "neurons": [25],
    "activation": ["relu"],
    "levels": [3],
}
TRAINING_PARAMS_cf = {
    "num_epochs": [5000],
    "batch_size": [265],
    "regularization_exp": [2],
    "regularization_param": [1e-6],
    "optimizer": ["ADAM"],
    "learning_rate": [0.001],
}
# # linear regression
# MODEL_PARAMS_cf = {
#     "input_dimension": [8],
#     "output_dimension": [1],
#     "n_hidden_layers": [5],
#     "neurons": [15],
#     "activation": ["relu"],
# }
# TRAINING_PARAMS_cf = {
#     "num_epochs": [1000],
#     "batch_size": [265],
#     "regularization_exp": [2],
#     "regularization_param": [1e-4],
#     "optimizer": ["LBFGS"],
#     "learning_rate": [0.001],
#     # seed:  1
# }
# # hiden
