import torch.utils.data
import itertools


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, input_width, label_width, offset=None):
        self.data = data

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
            input_data = self.data[index: index + self.input_width]
            label_data = self.data[index + self.step_1: index + self.step_2]
        elif self.__len__() <= index < self.max_len:
            input_data = self.data[index: index + self.input_width]
            label_data = torch.zeros([self.label_width])
        elif self.max_len <= index:
            raise IndexError('list index out of range')
        return input_data, label_data

    def __len__(self):
        return self.data.size(0) - self.offset - self.input_width


def get_structured_prediction(model, data_input, sequence_stride=None):
    if sequence_stride is None:
        sequence_stride = data_input.input_width
    step_1 = data_input.input_width + data_input.offset - data_input.label_width
    step_2 = data_input.input_width + data_input.offset

    # TODO: these indices could be specified further
    data_indices = [i for i in range(0, data_input.max_len, sequence_stride)]
    pred_time_indices = [ti for i in data_indices for ti in range(i + step_1, i + step_2)]
    input_data_subset = torch.utils.data.Subset(data_input, data_indices)
    input_data_ = torch.utils.data.DataLoader(input_data_subset, batch_size=len(input_data_subset), shuffle=False)
    input_data, _ = next(iter(input_data_))
    data_pred = model(input_data).detach()

    # returns time indices and predictions for those indices
    return pred_time_indices, torch.flatten(data_pred)
