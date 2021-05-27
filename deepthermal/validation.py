import torch
import torch.utils
from torch.utils.data import Subset, DataLoader
import itertools
from sklearn.model_selection import KFold


# Root Relative Squared Error
def get_RRSE(model, data, type_str="", verbose=False):
    # Compute the relative mean square error
    x_data, y_data = next(iter(DataLoader(data, batch_size=len(data), shuffle=False)))
    y_pred = model(x_data).detach()
    y_data_mean = torch.mean(y_data, dim=0)
    relative_error_2 = torch.sum((y_pred - y_data) ** 2) / torch.sum((y_data_mean - y_data) ** 2)
    relative_error = relative_error_2 ** 0.5
    if verbose: print(f"Root Relative Squared {type_str} Error: ", relative_error.item() * 100, "%")
    return relative_error.item()


# Normalized root-mean-square error
def get_NRMSE(model, data, type_str="", verbose=False):
    # Compute the relative mean square error
    x_data, y_data = next(iter(DataLoader(data, batch_size=len(data), shuffle=False)))
    y_pred = model(x_data).detach()
    error = torch.mean((y_pred - y_data) ** 2) ** 0.5
    norm_error = torch.mean((y_pred - y_data) ** 2) ** 0.5 / torch.mean(y_data)
    if verbose: print(f"Root mean square {type_str} error: ", error.item() * 100, "%")
    return error.item()


def k_fold_cv_grid(Model, model_param_iter, fit, training_param_iter, data, folds=5, init=None, partial=False,
                   verbose=False):
    models = []
    loss_history_trains = []
    loss_history_vals = []
    rel_train_errors = []
    rel_val_errors = []
    for model_num, (model_param, training_param) in enumerate(itertools.product(model_param_iter, training_param_iter)):
        kf = KFold(n_splits=folds, shuffle=True)
        rel_train_errors_k = []
        rel_val_errors_k = []
        models_k = []
        loss_history_trains_k = []
        loss_history_vals_k = []
        for k_num, (train_index, val_index) in enumerate(kf.split(data)):
            if verbose: print(f"Running model (mod={model_num},k={k_num})")
            data_train_k = Subset(data, train_index)
            data_val_k = Subset(data, val_index)
            model = Model(**model_param)
            if init is not None:
                init(model, **training_param)
            loss_history_train, loss_history_val = fit(model, **training_param, data=data_train_k, data_val=data_val_k)

            models_k.append(model)
            loss_history_trains_k.append(loss_history_train)
            loss_history_vals_k.append(loss_history_val)
            rel_train_errors_k.append(get_RRSE(model, data_train_k))
            rel_val_errors_k.append(get_RRSE(model, data_val_k))

            if partial:
                break

        models.append(models_k)
        loss_history_trains.append(loss_history_trains_k)
        loss_history_vals.append(loss_history_vals_k)
        rel_train_errors.append(rel_train_errors_k)
        rel_val_errors.append(rel_val_errors_k)

    k_fold_grid = {
        "models": models,
        "loss_history_trains": loss_history_trains,
        "loss_history_vals": loss_history_vals,
        "rel_train_errors": rel_train_errors,
        "rel_val_errors": rel_val_errors
    }
    return k_fold_grid


def create_subdictionary_iterator(dictionary):
    for sublist in itertools.product(*dictionary.values()):
        yield dict(zip(dictionary.keys(), sublist))


# printing model errors
def print_model_errors(rel_val_errors, **kwargs):
    for i, rel_val_error_list in enumerate(rel_val_errors):
        avg_error = sum(rel_val_error_list) / len(rel_val_error_list)
        print(f"Model {i} Root Relative Squared validation error: {avg_error * 100}%")
