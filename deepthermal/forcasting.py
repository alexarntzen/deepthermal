import torch.utils.data


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, input_width, label_width, offset=None):
        self.data = data

        self.input_width = input_width
        self.label_width = label_width
        if offset is None:
            self.offset = label_width
        else:
            self.offset = label_width

    def __getitem__(self, index):
        input_data = self.data[index: index + self.input_width]
        label_data = self.data[index + self.input_width + self.offset - self.input_width:
                               index + self.input_width + self.offset]
        return input_data, label_data

    def __len__(self):
        return self.data.size(0) - self.offset
