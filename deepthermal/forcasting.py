import torch
from torch.utils.data import Dataset


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

        self.max_len = self.data.size(0) - input_width
        self.step_1 = self.input_width + self.offset - self.label_width
        self.step_2 = self.input_width + self.offset

    def __getitem__(self, index):
        if index < self.__len__():
            input_data = self.data[index : index + self.input_width]
            label_data = self.data[index + self.step_1 : index + self.step_2]
        elif self.__len__() <= index < self.max_len:
            input_data = self.data[index : index + self.input_width]
            label_data = torch.zeros_like(self.data)[self.step_1 : self.step_2]
        elif self.max_len <= index:
            raise IndexError("list index out of range")
        return input_data, label_data

    def __len__(self):
        return self.data.size(0) - self.offset - self.input_width


def get_structured_prediction(model, data_input, sequence_stride=None):
    if sequence_stride is None:
        sequence_stride = data_input.label_width

    # TODO: these indices could be specified further
    # this weird sequence construction is made so that we allways include the last element
    data_indices = [i for i in range(data_input.max_len - 1, -1, -sequence_stride)][
        ::-1
    ]
    pred_time_indices = [
        ti
        for i in data_indices
        for ti in range(i + data_input.step_1 + 1, i + data_input.step_2 + 1)
    ]

    input_data_subset = torch.utils.data.Subset(data_input, data_indices)
    input_data_ = torch.utils.data.DataLoader(
        input_data_subset, batch_size=len(input_data_subset), shuffle=False
    )
    input_data, _ = next(iter(input_data_))
    data_pred = model(input_data).detach()

    # returns time indices and predictions for those indices
    return pred_time_indices, torch.flatten(data_pred)
