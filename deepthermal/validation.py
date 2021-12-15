import torch
import torch.utils
from torch.utils.data import Subset, DataLoader, Dataset

import itertools
from sklearn.model_selection import KFold

from deepthermal.FFNN_model import get_scaled_model, fit_FFNN


# Root Relative Squared Error
def get_RRSE(model, data, type_str="", verbose=False):
    # Compute the relative mean square error
    x_data, y_data = next(iter(DataLoader(data, batch_size=len(data), shuffle=False)))
    y_pred = model(x_data).detach()
    y_data_mean = torch.mean(y_data, dim=0)
    relative_error_2 = torch.sum((y_pred - y_data) ** 2) / torch.sum(
        (y_data_mean - y_data) ** 2
    )
    relative_error = relative_error_2 ** 0.5
    if verbose:
        print(
            f"Root Relative Squared {type_str} Error: ",
            relative_error.item() * 100,
            "%",
        )
    return relative_error.item()


# Normalized root-mean-square error
def get_NRMSE(model, data, type_str="", verbose=False):
    # Compute the relative mean square error
    x_data, y_data = next(iter(DataLoader(data, batch_size=len(data), shuffle=False)))
    y_pred = model(x_data).detach()
    error = torch.mean((y_pred - y_data) ** 2) ** 0.5
    if verbose:
        print(f"Root mean square {type_str} error: ", error.item() * 100, "%")
    return error.item()


def k_fold_cv_grid(
    model_params,
    training_params,
    data: Dataset,
    val_data: Dataset = None,
    fit: callable = fit_FFNN,
    folds=5,
    trials=1,
    partial=False,
    verbose=False,
    get_error=None,
):
    # transform a dictionary with a list of inputs
    # into an iterator over the coproduct of the lists
    if isinstance(model_params, dict):
        model_params_iter = create_subdictionary_iterator(model_params)
    else:
        model_params_iter = model_params

    if isinstance(training_params, dict):
        training_params_iter = create_subdictionary_iterator(training_params)
    else:
        training_params_iter = training_params

    models = []
    loss_history_trains = []
    loss_history_vals = []
    rel_train_errors = []
    rel_val_errors = []
    for model_num, (trial, (model_param, training_param)) in enumerate(
        itertools.product(
            range(trials), itertools.product(model_params_iter, training_params_iter)
        )
    ):

        splits = (
            KFold(n_splits=folds, shuffle=True).split(data)
            if folds > 1
            else ((None, []),)  # ((full set, empty set),)
        )

        rel_train_errors_k = []
        rel_val_errors_k = []
        models_instance_k = []
        loss_history_trains_k = []
        loss_history_vals_k = []
        for k_num, (train_index, val_index) in enumerate(splits):
            if verbose:
                print(f"\nRunning model (trial={trial}, mod={model_num}, k={k_num}):")
                print(f"Parameters: {model_param, training_param}")
            data_train_k = data if train_index is None else Subset(data, train_index)

            # if no validaton data do k_fold splits
            if val_data is None:
                data_val_k = data if val_index is None else Subset(data, val_index)
            else:
                data_val_k = val_data
            # train model on data!
            model_param_k = model_param.copy()
            model_instance = model_param_k.pop("model")(**model_param_k)
            model_instance, loss_history_train, loss_history_val = fit(
                model=model_instance,
                **training_param,
                data=data_train_k,
                data_val=data_val_k,
                verbose=verbose,
            )

            models_instance_k.append(model_instance)
            loss_history_trains_k.append(loss_history_train)
            loss_history_vals_k.append(loss_history_val)
            if callable(get_error):
                rel_train_errors_k.append(get_error(model_instance, data_train_k))
                rel_val_errors_k.append(get_error(model_instance, data_val_k))
            if partial:
                break

        models.append(models_instance_k)
        loss_history_trains.append(loss_history_trains_k)
        loss_history_vals.append(loss_history_vals_k)
        if len(rel_train_errors_k) > 0:
            rel_train_errors.append(rel_train_errors_k)
            rel_val_errors.append(rel_val_errors_k)

    k_fold_grid = {
        "models": models,
        "loss_history_trains": loss_history_trains,
        "loss_history_vals": loss_history_vals,
        "rel_train_errors": rel_train_errors,
        "rel_val_errors": rel_val_errors,
    }
    return k_fold_grid


def create_subdictionary_iterator(dictionary: dict, product=True) -> iter:
    """Create an iterator over a dictionary of lists
    Important: all lists in the dict must be of the same lenght if zip is chosen
    Cartesian product is default
    """
    combine = itertools.product if product else zip
    for sublist in combine(*dictionary.values()):
        # convert two list into dictionary
        yield dict(zip(dictionary.keys(), sublist))


def add_dictionary_iterators(new_dict_iter: iter, dict_iterator: iter, product=True):
    """Combine two subdictionary iterators"""
    combine = itertools.product if product else zip
    for left, right in combine(new_dict_iter, dict_iterator):
        yield right | left


# printing model errors
def print_model_errors(rel_val_errors, **kwargs):
    for i, rel_val_error_list in enumerate(rel_val_errors):
        avg_error = sum(rel_val_error_list) / len(rel_val_error_list)
        print(f"Model {i} average error: {avg_error}")


def get_scaled_results(cv_results, x_center=0, x_scale=1, y_center=0, y_scale=1):
    cv_results_scaled = cv_results.copy()
    cv_results_scaled["models"] = []
    for i in range(len(cv_results["models"])):
        cv_results_scaled["models"].append([])
        for j in range(len(cv_results["models"][i])):
            cv_results_scaled["models"][i].append(
                get_scaled_model(
                    cv_results["models"][i][j],
                    x_center=x_center,
                    x_scale=x_scale,
                    y_center=y_center,
                    y_scale=y_scale,
                )
            )
    return cv_results_scaled
