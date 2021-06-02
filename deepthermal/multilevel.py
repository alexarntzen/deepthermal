import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from deepthermal.FFNN_model import fit_FFNN, init_xavier, FFNN


# this is not the most effective way, but it is easy
class MultilevelDataset(Dataset):
    def __init__(self, y_tensors, x_tensor):
        # last tensor are the x_values

        self.levels = len(y_tensors)
        rows = x_tensor.size(0)
        new_y_tensor = [y_tensors[0]]
        y_nan = torch.full_like(y_tensors[0], float("nan"))
        assert x_tensor.size(0) == y_tensors[0].size(0), "Tensor lenght of y ground level does not mach x"
        self.level_len = torch.zeros(self.levels, dtype=torch.int)
        self.level_len[0] = y_tensors[0].size(0)
        for i in range(self.levels - 1):
            assert y_tensors[i].size(0) >= y_tensors[i + 1].size(0), "First tensor should be one with the most data"
            self.level_len[i + 1] = y_tensors[i + 1].size(0)
            new_y_tensor.append(torch.cat((y_tensors[i + 1], y_nan), dim=0)[:rows])
        self.y_tensors = new_y_tensor
        self.x_tensor = x_tensor

    def __getitem__(self, index):
        return self.x_tensor[index], tuple(tensor[index] for tensor in self.y_tensors)

    def __len__(self):
        return self.x_tensor.size(0)


def get_level_dataset(x_tensor, y_tensors, level):
    if level == 0:
        return x_tensor, y_tensors[0]
    elif level > 0:
        # not the most efficient, I do not care right now
        test = ~torch.isnan(y_tensors[level])

        level_indices = test.nonzero()[:, 0]
        diff = y_tensors[level][level_indices] - y_tensors[level - 1][level_indices]
        return x_tensor[level_indices], diff


def fit_multilevel_FFNN(models,
                        data,
                        num_epochs,
                        batch_size,
                        optimizer,
                        p=2,
                        regularization_param=0,
                        regularization_exp=2,
                        data_val=None,
                        track_history=True,
                        verbose=False,
                        learning_rate=None,
                        **kwargs
                        ):
    levels = len(models)
    loss_history_train_levels = torch.zeros((levels, num_epochs))
    loss_history_val_levels = torch.zeros((levels, num_epochs))
    for level in range(len(models)):
        level_data = TensorDataset(*get_level_dataset(*data[:], level))
        loss_history_train_levels[level], loss_history_val_levels[level] = fit_FFNN(model=models[level],
                                                                                    data=level_data,
                                                                                    num_epochs=num_epochs,
                                                                                    batch_size=batch_size,
                                                                                    optimizer=optimizer,
                                                                                    p=p,
                                                                                    regularization_param=regularization_param,
                                                                                    regularization_exp=regularization_exp,
                                                                                    data_val=data_val,
                                                                                    track_history=track_history,
                                                                                    verbose=verbose,
                                                                                    learning_rate=learning_rate
                                                                                    )

    # return the sum of the losses since it is not relative loss
    loss_history_train = torch.sum(loss_history_train_levels, dim=0)
    loss_history_val = torch.sum(loss_history_val_levels, dim=0)

    return loss_history_train, loss_history_val


def predict_multilevel(models, x_data):
    y_pred = models[0](x_data)
    for level in range(1, len(models)):
        y_pred += models[level](x_data)
    return y_pred

def get_trained_multilevel_model(model_param, training_param,
                                 multilevel_data,
                                 data_val=None,
                                 Model=FFNN,
                                 fit=fit_multilevel_FFNN,
                                 init=init_xavier
                                 ):
    models = []
    for l in range(multilevel_data.levels):
        model = Model(**model_param)
        # Xavier weight initialization
        init(model, **training_param)
        models.append(model)
    loss_history_train, loss_history_val = fit(
        models, multilevel_data, data_val=data_val, **training_param
    )
    return models, loss_history_train, loss_history_val
