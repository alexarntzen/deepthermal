import torch
import torch.utils
import torch.utils.data
import itertools
from sklearn.model_selection import KFold


def get_rMSE(model, x, y, type_str="", verbose=False):
    # Compute the relative mean square error
    y_pred = model(x)
    relative_error = torch.mean((y_pred - y) ** 2) / torch.mean(y ** 2)
    if verbose: print(f"Relative {type_str} error: ", relative_error.item() ** 0.5 * 100, "%")
    return relative_error.item()


def k_fold_CV_grid(Model, model_param_iter, fit, training_param_iter, x, y, folds=5, init=None, partial=False,
                   verbose=False):
    models = []
    rel_train_errors = []
    rel_val_errors = []
    for model_num, (model_param, training_param) in enumerate(itertools.product(model_param_iter, training_param_iter)):
        kf = KFold(n_splits=folds, shuffle=True)
        rel_train_errors_k = []
        rel_val_errors_k = []
        models_k = []
        for k_num, (train_index, val_index) in enumerate(kf.split(x)):
            if verbose: print(f"Running model (mod={model_num},k={k_num})")
            x_train_k, x_val_k = x[train_index], x[val_index]
            y_train_k, y_val_k = y[train_index], y[val_index]

            model = Model(**model_param)
            if init is not None: init(model, **training_param)

            fit(model, x_train_k, y_train_k, **training_param, x_val=x_val_k, y_val=y_val_k)
            models_k.append(model)
            rel_train_errors_k.append(get_rMSE(model, x_train_k, y_train_k))
            rel_val_errors_k.append(get_rMSE(model, x_val_k, y_val_k))

            if partial: break

        rel_train_errors.append(rel_train_errors_k)
        rel_val_errors.append(rel_val_errors_k)
        models.append(models_k)

    return models, rel_train_errors, rel_val_errors


def create_subdictionary_iterator(dictionary):
    for sublist in itertools.product(*dictionary.values()):
        yield dict(zip(dictionary.keys(), sublist))

