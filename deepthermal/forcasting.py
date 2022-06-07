import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_width, label_width, offset=None):
        self.data = data
        self.rest_shape = data.shape[1:]
        self.input_width = input_width
        self.label_width = label_width
        if offset is None:
            self.offset = label_width
        else:
            self.offset = offset

        self.max_len = self.data.size(0) - input_width + 1
        self.step_1 = self.input_width + self.offset - self.label_width
        self.step_2 = self.input_width + self.offset

    def __getitem__(self, index):
        if (
            (isinstance(index, np.ndarray) or isinstance(index, torch.Tensor))
            and len(index.shape)
        ) > 0 or isinstance(index, list):
            inputs, labels = [], []
            for i in index:
                input, label = self.__getitem__(i)
                inputs.append(input)
                labels.append(label)
            return torch.stack(inputs), torch.stack(labels)
        else:
            if index < 0:
                raise IndexError("list index out of range")
            elif index < self.__len__():
                input_data = self.data[index : index + self.input_width]
                label_data = self.data[index + self.step_1 : index + self.step_2]
                return input_data, label_data
            elif self.__len__() <= index < self.max_len:
                input_data = self.data[index : index + self.input_width]
                label_data = torch.zeros_like(self.data)[self.step_1 : self.step_2]
                return input_data, label_data
            elif self.max_len <= index:
                raise IndexError("list index out of range")

    def __len__(self):
        return self.data.size(0) - self.offset - (self.input_width - 1)


def get_structured_prediction(
    model, data_input, sequence_stride=None, prediction_only=False
):
    if sequence_stride is None:
        sequence_stride = data_input.label_width

    # TODO: these indices could be specified further
    # this weird sequence construction is made so that
    # we allways include the last element
    if prediction_only:
        data_indices = [data_input.max_len - 1]
    else:
        data_indices = [i for i in range(data_input.max_len - 1, -1, -sequence_stride)][
            ::-1
        ]
    pred_time_indices = [
        ti
        for i in data_indices
        for ti in range(i + data_input.step_1, i + data_input.step_2)
    ]

    input_data_subset = Subset(data_input, data_indices)
    input_data_ = DataLoader(
        input_data_subset, batch_size=len(input_data_subset), shuffle=False
    )
    input_data, _ = next(iter(input_data_))
    data_pred = model(input_data).detach()
    # returns time indices and predictions for those indices
    return pred_time_indices, torch.flatten(data_pred, 0, 1)


class LSTM(nn.Module):
    def __init__(
        self, input_dimension, output_dimension, neurons, num_layers=1, label_width=None
    ):
        super().__init__()
        self.label_width = label_width
        self.hidden_layer_size = neurons
        self.lstm = nn.LSTM(
            input_dimension, neurons, batch_first=True, num_layers=num_layers
        )

        self.linear = nn.Linear(neurons, output_dimension)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out)
        if self.label_width is not None:
            return predictions[:, -self.label_width :]
        else:
            return predictions
