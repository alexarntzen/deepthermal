import matplotlib.pyplot as plt
import torch


def get_disc_str(model):
    params = {"activation": model.activation, "n_hidden_layers": model.n_hidden_layers, "neurons": model.neurons,
              "epochs": len(model.loss_history_train)}
    return str(params)


def plot_model_history(models, plot_name="0", path_figures="figures"):
    k = len(models)
    histfig, axis = plt.subplots(1, k, figsize=(8 * k, 6))
    for i, model in enumerate(models):
        axis[i].grid(True, which="both", ls=":")
        axis[i].plot(torch.arange(1, len(model.loss_history_train) + 1), model.loss_history_train,
                label="Training error history", )
        if len(model.loss_history_val) > 0:
            axis[i].plot(torch.arange(1, len(model.loss_history_val) + 1), model.loss_history_val,
                    label="Validation error history")
        axis[i].set_xlabel("Epoch")
        axis[i].set_ylabel("Error")
        axis[i].set_yscale("log")
        axis[i].legend()
        histfig.suptitle(f"History, model: {get_disc_str(model)}")
    histfig.savefig(f"{path_figures}/history_{plot_name}.pdf")
    plt.close(histfig)


# plot visualization
def plot_model_1d(model, x_test, plot_name="vis_model", x_train=None, y_train=None, path_figures="../figures"):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Model: {get_disc_str(model)}")
    for i in range(model.output_dimension):
        if x_train is not None and y_train is not None:
            ax.scatter(x_train[:, 0], y_train[:, i], label=f"train_{i}")
        ax.plot(x_test[:, 0], model(x_test)[:, i].detach(), label=f"pred_{i}", lw=2, ls="-.")
    ax.legend()
    fig.savefig(f"{path_figures}/{plot_name}.pdf")
    plt.close(fig)
