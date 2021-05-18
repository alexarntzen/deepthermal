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


def k_fold_cv_grid(Model, model_param_iter, fit, training_param_iter, x, y, folds=5, init=None, partial=False,
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
        for k_num, (train_index, val_index) in enumerate(kf.split(x)):
            if verbose: print(f"Running model (mod={model_num},k={k_num})")
            x_train_k, x_val_k = x[train_index], x[val_index]
            y_train_k, y_val_k = y[train_index], y[val_index]

            model = Model(**model_param)
            if init is not None: init(model, **training_param)

            loss_history_train, loss_history_val = fit(model, x_train_k, y_train_k, **training_param, x_val=x_val_k,
                                                       y_val=y_val_k)

            models_k.append(model)
            loss_history_trains_k.append(loss_history_train)
            loss_history_vals_k.append(loss_history_val)
            rel_train_errors_k.append(get_rMSE(model, x_train_k, y_train_k))
            rel_val_errors_k.append(get_rMSE(model, x_val_k, y_val_k))

            if partial: break

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
