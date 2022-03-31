import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch

from deepthermal.validation import print_model_errors


def get_disc_str(model):
    params = {
        "activation": model.activation,
        "n_hidden_layers": model.n_hidden_layers,
        "neurons": model.neurons,
    }
    return str(params)


def plot_model_history(
    models,
    loss_history_trains,
    loss_history_vals=None,
    plot_name="Loss history",
    path_figures="figures",
):

    if len(loss_history_trains) == 0:
        warnings.warn("No loss history to plot")
        return

    k = len(models)
    histfig, axis = plt.subplots(1, k, tight_layout=True)
    if k == 1:
        axis = [axis]
    for i, model in enumerate(models):
        if np.any(loss_history_trains[i] < 0) or (
            loss_history_vals is not None and np.any(loss_history_vals[i] < 0)
        ):
            plot_func = axis[i].plot
        else:
            plot_func = axis[i].loglog

        plot_func(
            torch.arange(1, len(loss_history_trains[i]) + 1),
            loss_history_trains[i],
            label="Training error history",
        )
        if loss_history_vals is not None and len(loss_history_vals[i]) > 0:
            plot_func(
                torch.arange(1, len(loss_history_vals[i]) + 1),
                loss_history_vals[i],
                label="Validation error history",
            )
        axis[i].set_xlabel("Iterations")
        axis[i].set_ylabel("Loss")
        axis[i].legend()
    histfig.savefig(f"{path_figures}/history_{plot_name}.pdf")
    plt.close(histfig)


# Todo: make this iterable over datsets
def plot_result_sorted(
    x_pred=None,
    y_pred=None,
    x_train=None,
    y_train=None,
    plot_name="",
    x_axis="",
    y_axis="",
    path_figures="../figures",
    compare_label="data",
    fig: Figure = None,
) -> Figure:
    if Figure is None:
        fig, ax = plt.subplots(tight_layout=True)
    else:
        ax = fig.get_axes()[0]

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    if x_train is not None and y_train is not None:
        ax.plot(x_train, y_train, ".:", label=compare_label, lw=2, mew=1)
    if x_pred is not None and y_pred is not None:
        ax.plot(x_pred, y_pred, "-", label="Prediction", lw=2)
    ax.legend()
    fig.savefig(f"{path_figures}/{plot_name}.pdf")
    plt.close(fig)
    return fig


# plot predicted data on
def plot_model_scatter(
    model,
    x_test,
    plot_name="vis_model",
    x_train=None,
    y_train=None,
    path_figures="../figures",
):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_ylabel(r"$y$")
    ax.set_xlabel(r"$||x||$")
    for i in range(y_train.size(-1)):
        if x_train is not None and y_train is not None:
            ax.scatter(
                torch.norm(x_train, p=2, dim=1),
                y_train[:, i],
                label=f"train_{i}",
                marker=".",
            )
        ax.scatter(
            torch.norm(x_test, p=2, dim=1),
            model(x_test)[:, i].detach(),
            label=f"pred_{i}",
            marker="x",
            alpha=0.5,
            color="r",
            lw=1,
        )
    ax.legend()
    fig.savefig(f"{path_figures}/{plot_name}.pdf")
    plt.close(fig)


# plot predicted data on
def plot_compare_scatter(
    model, x_train, y_train, plot_name="vis_model", path_figures="../figures", **kwargs
):
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xlabel("Actual data")
    ax.set_ylabel("Predicted data")
    for i in range(y_train.size(-1)):
        ax.scatter(
            y_train[:, i],
            model(x_train).detach()[:, i],
            label=f"pred nr. {i}",
            marker=".",
            lw=1,
        )
    ax.legend()
    fig.savefig(f"{path_figures}/{plot_name}.pdf")
    plt.close(fig)


# plot visualization
def plot_model_1d(model, x_test, **kwargs):
    y_pred = model(x_test).detach()
    plot_result_sorted(y_pred=y_pred, x_pred=x_test, **kwargs)


def plot_result(
    models,
    loss_history_trains,
    loss_history_vals,
    path_figures="",
    plot_name="plot",
    rel_val_errors=None,
    plot_function=None,
    function_kwargs=None,
    model_list=None,
    history=True,
    **kwargs,
):
    if model_list is None:
        model_list = np.arange(len(models))
    if rel_val_errors is not None:
        print_model_errors(rel_val_errors)
    for i in model_list:
        if history:
            plot_model_history(
                models[i],
                loss_history_trains[i],
                loss_history_vals[i],
                plot_name=f"{plot_name}_{i}",
                path_figures=path_figures,
            )
        if plot_function is not None:
            for j in range(len(models[i])):
                plot_function(
                    plot_name=f"{plot_name}_{i}_{j}",
                    model=models[i][j],
                    path_figures=path_figures,
                    **function_kwargs,
                )
