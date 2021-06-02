import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from deepthermal.FFNN_model import fit_FFNN, init_xavier, FFNN


class MultilevelFFNN(FFNN):
    def __init__(self, levels=1, **model_params):
        self.levels = levels
        self.models = []
        for level in range(levels):
            self.models.append(FFNN(**model_params))
        super().__init__(**model_params)

    def __getitem__(self, item):
        return self.models[item]

    def __call__(self, x_data):
        y_pred = self.models[0](x_data)
        for level in range(1, len(self.models)):
            y_pred += self.models[level](x_data)
        return y_pred

    def __len__(self):
        return self.levels


# this is not the most effective way, but it is easy
class MultilevelDataset(Dataset):
    def __init__(self, x_tensor, y_tensors):
        # last tensor are the x_values

        self.levels = len(y_tensors)
        rows = x_tensor.size(0)
        new_y_tensor = [y_tensors[0]]
        y_nan = torch.full_like(y_tensors[0], float("nan"))
        assert x_tensor.size(0) == y_tensors[0].size(
            0
        ), "Tensor lenght of y ground level does not mach x"
        self.level_len = torch.zeros(self.levels, dtype=torch.int)
        self.level_len[0] = y_tensors[0].size(0)
        for i in range(self.levels - 1):
            assert y_tensors[i].size(0) >= y_tensors[i + 1].size(
                0
            ), "First tensor should be one with the most data"
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


def fit_multilevel_FFNN(Models, data, data_val=None, num_epochs=100, **training_param):
    levels = len(Models)
    loss_history_train_levels = torch.zeros((levels, num_epochs))
    loss_history_val_levels = torch.zeros((levels, num_epochs))
    for level in range(len(Models)):
        level_data = TensorDataset(*get_level_dataset(*data[:], level))
        if data_val is not None:
            level_data_val = TensorDataset(*get_level_dataset(*data_val[:], level))
        else:
            level_data_val = None
        loss_history_train_levels[level], loss_history_val_levels[level] = fit_FFNN(
            model=Models[level],
            data=level_data,
            data_val=level_data_val,
            num_epochs=num_epochs,
            **training_param,
        )

    # return the sum of the losses since it is not relative loss
    loss_history_train = torch.sum(loss_history_train_levels, dim=0)
    loss_history_val = torch.sum(loss_history_val_levels, dim=0)

    return loss_history_train, loss_history_val


def get_init_multilevel(init=init_xavier):
    def init_multilevel(Models, **kwargs):
        for level in range(len(Models)):
            init(Models[level], **kwargs)

    return init_multilevel


# Root Relative Squared Error
def get_multilevel_RRSE(model, data, type_str="", verbose=False, level=0):
    # Compute the relative mean square error
    x_data, y_data_list = next(
        iter(DataLoader(data, batch_size=len(data), shuffle=False))
    )
    y_data = y_data_list[level]
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


def predict_multilevel(models, x_data):
    y_pred = models[0](x_data)
    for level in range(1, len(models)):
        y_pred += models[level](x_data)
    return y_pred
