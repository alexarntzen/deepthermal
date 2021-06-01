import torch
from torch.utils.data.dataset import Dataset

import torch
from torch.utils.data import Dataset


# this is not the most effective way, but it is easy
class MultilevelDataset(Dataset):
    def __init__(self, y_tensors, x_tensor):
        # last tensor are the x_values

        l = x_tensor.size(0)
        new_y_tensor = [y_tensors[0]]
        y_nan = torch.full_like(y_tensors[0], float("nan"))
        assert x_tensor.size(0) == y_tensors[0].size(0), "Tensor lenght of y ground level does not mach x"
        self.level_len = torch.zeros(len(y_tensors), dtype=torch.int)
        self.level_len[0] = y_tensors[0].size(0)
        for i in range(len(y_tensors) - 1):
            assert y_tensors[i].size(0) >= y_tensors[i + 1].size(0), "First tensor should be one with the most data"
            self.level_len[i + 1] = y_tensors[i + 1].size(0)
            new_y_tensor.append(torch.cat((y_tensors[i + 1], y_nan), dim=0)[:l])
        self.y_tensors = new_y_tensor
        self.x_tensor = x_tensor

    def __getitem__(self, index):
        return self.x_tensor[index], tuple(tensor[index] for tensor in self.y_tensors)

    def __len__(self):
        return self.x_tensor.size(0)


def get_level_dataset(x_tensor, y_tensor, level):
    if level == 0:
        return x_tensor, y_tensor[0]
    elif level > 0:
        # not the most efficient, I do not care right now
        test = ~torch.isnan(y_tensor[level])

        level_indices = test.nonzero()[:, 0]
        diff = y_tensor[level][level_indices] - y_tensor[level - 1][level_indices]
        return x_tensor[level_indices], diff

#
#
# def fit_multilevel(
#         model,
#         data,
#         num_epochs,
#         batch_size,
#         optimizer,
#         p=2,
#         regularization_param=0,
#         regularization_exp=2,
#         data_val=None,
#         track_history=True,
#         verbose=False,
#         learning_rate=None,
#         **kwargs
# ):
#     training_set = DataLoader(data, batch_size=batch_size, shuffle=True)
#     if learning_rate is None:
#         learning_rate = larning_rates[optimizer]
#     # select optimizer
#     if optimizer == "ADAM":
#         optimizer_ = optim.Adam(model.parameters(), lr=learning_rate)
#     elif optimizer == "LBFGS":
#         optimizer_ = optim.LBFGS(
#             model.parameters(),
#             lr=learning_rate,
#             max_iter=1,
#             max_eval=50000,
#             tolerance_change=1.0 * np.finfo(float).eps,
#         )
#     else:
#         raise ValueError("Optimizer not recognized")
#
#     loss_history_train = np.zeros((num_epochs))
#     loss_history_val = np.zeros((num_epochs))
#     # Loop over epochs
#     for epoch in range(num_epochs):
#         if verbose:
#             print(
#                 "################################ ",
#                 epoch,
#                 " ################################",
#             )
#
#         # Loop over batches
#         for j, (x_train_, y_train_) in enumerate(training_set):
#
#             def closure():
#                 # zero the parameter gradients
#                 optimizer_.zero_grad()
#                 # forward + backward + optimize
#                 y_pred_ = model(x_train_)
#                 loss_u = torch.mean((y_pred_ - y_train_) ** p)
#                 loss_reg = regularization(model, regularization_exp)
#                 loss = loss_u + regularization_param * loss_reg
#                 loss.backward()
#
#                 # Compute average training loss over batches for the current epoch
#                 if track_history:
#                     loss_history_train[epoch] += loss.item() / len(training_set)
#                 return loss
#
#             optimizer_.step(closure=closure)
#
#         if data_val:
#             x_val, y_val = next(
#                 iter(DataLoader(data_val, batch_size=len(data_val), shuffle=False))
#             )
#         # record validation loss for history
#         if data_val is not None and track_history:
#             y_val_pred_ = model(x_val)
#             validation_loss = torch.mean((y_val_pred_ - y_val) ** p).item()
#             loss_history_val[epoch] = validation_loss
#
#         if verbose and track_history:
#             print("Training Loss: ", np.round(loss_history_train[-1], 8))
#             if data_val is not None:
#                 print("Validation Loss: ", np.round(validation_loss, 8))
#
#     if verbose and track_history:
#         print("Final Training Loss: ", np.round(loss_history_train[-1], 8))
#         if data_val is not None:
#             print("Final Validation Loss: ", np.round(loss_history_val[-1], 8))
#
#     return loss_history_train, loss_history_val


def predict_multilevel(model, data_input, sequence_stride=None):
    pass
    # if sequence_stride is None:
    #     sequence_stride = data_input.label_width
    #
    # # TODO: these indices could be specified further
    # # this weird sequence construction is made so that we allways include the last element
    # data_indices = [i for i in range(data_input.max_len - 1, -1, -sequence_stride)][
    #                ::-1
    #                ]
    # pred_time_indices = [
    #     ti
    #     for i in data_indices
    #     for ti in range(i + data_input.step_1 + 1, i + data_input.step_2 + 1)
    # ]
    #
    # input_data_subset = torch.utils.data.Subset(data_input, data_indices)
    # input_data_ = torch.utils.data.DataLoader(
    #     input_data_subset, batch_size=len(input_data_subset), shuffle=False
    # )
    # input_data, _ = next(iter(input_data_))
    # data_pred = model(input_data).detach()
    #
    # # returns time indices and predictions for those indices
    # return pred_time_indices, torch.flatten(data_pred)
