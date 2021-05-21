import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, input_width, label_width, offset=None):
        self.data = data
        
        self.input_width = input_width
        self.label_width = label_width
        if  offset is None: self.offset = label_width
        else: self.offset = label_width

    def __getitem__(self, index):
        input_data = self.data[index: index + self.input_width]
        label_data = self.data[index + self.input_width + self.offset - self.input_width: index + self.input_width + self.offset]
        return input_data, label_data

    def __len__(self):
       return self.data[0].size(0) - self.offset 

def fit_forcaster(model, data, num_epochs, batch_size, optimizer, p=2, regularization_param=0,
             regularization_exp=2, x_val=None,
             y_val=None, track_history=True, verbose=False, learning_rate=None, **kwargs):

    training_set = DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                              shuffle=False)
    #torch.utils.data.SequentialSampler(replacement=False):




    learning_rate = larning_rates[optimizer]
    # select optimizer
    if optimizer == "ADAM":
        optimizer_ = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "LBFGS":
        optimizer_ = optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=1, max_eval=50000,
                                 tolerance_change=1.0 * np.finfo(float).eps)
    else:
        raise ValueError("Optimizer not recognized")

    loss_history_train = np.zeros((num_epochs))
    loss_history_val = np.zeros((num_epochs))
    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose: print("################################ ", epoch, " ################################")

        # Loop over batches
        for j, (x_train_, u_train_) in enumerate(training_set):
            def closure():
                # zero the parameter gradients
                optimizer_.zero_grad()
                # forward + backward + optimize
                u_pred_ = model(x_train_)
                loss_u = torch.mean((u_pred_ - u_train_) ** p)
                loss_reg = regularization(model, regularization_exp)
                loss = loss_u + regularization_param * loss_reg
                loss.backward()

                # Compute average training loss over batches for the current epoch
                if track_history:
                    loss_history_train[epoch] += loss.item() / len(training_set)
                return loss

            optimizer_.step(closure=closure)

        # record validation loss for history
        if y_val is not None and track_history:
            y_val_pred_ = model(x_val)
            validation_loss = torch.mean((y_val_pred_ - y_val) ** p).item()
            loss_history_val[epoch] = validation_loss

        if verbose and track_history:
            print('Training Loss: ', np.round(loss_history_train[-1], 8))
            if y_val is not None: print('Validation Loss: ', np.round(validation_loss, 8))

    if verbose and track_history:
        print('Final Training Loss: ', np.round(loss_history_train[-1], 8))
        if y_val is not None: print('Final Validation Loss: ', np.round(loss_history_val[-1], 8))

    return loss_history_train, loss_history_val