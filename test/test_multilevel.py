import unittest
import torch
import torch.utils
import torch.utils.data

from deepthermal.multilevel import MultilevelDataset, get_level_dataset


class TestPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n_samples = 2000
        cls.x = torch.rand((cls.n_samples, 2))
        cls.y_0 = torch.rand((cls.n_samples, 1))
        cls.y_1 = cls.y_0[:cls.n_samples // 2]
        cls.y_2 = cls.y_0[:cls.n_samples // 4]
        cls.datalist = [cls.y_0.detach().clone(), cls.y_1.detach().clone(), cls.y_2.detach().clone()]
        print(cls.datalist[1].shape)
        cls.len_list = [cls.n_samples, cls.n_samples // 2, cls.n_samples // 4]

    def test_dataset_params(self):
        print("\n\n Test whether MultilevelDataset stores the right data and return the right dataset:")
        dataset = MultilevelDataset(self.datalist, self.x)

        x_data, y_data = dataset[:]
        for l in range(len(y_data)):
            self.assertEqual(x_data.size(0), y_data[l].size(0))
            self.assertEqual(torch.sum(torch.isnan(y_data[l])), self.n_samples - self.len_list[l])
            x_train, y_train = get_level_dataset(x_data, y_data, l)
            self.assertEqual(x_train.size(0), y_train.size(0))
            self.assertEqual(torch.max(torch.abs(self.x[:self.len_list[l]] - x_train)).item(), 0)
            if l == 0:
                self.assertEqual(torch.max(torch.abs(y_train - self.y_0)).item(), 0)
            elif l > 0:
                self.assertEqual(
                    torch.max(torch.abs(
                        y_train - self.datalist[l] + self.datalist[l - 1][:self.len_list[l]]
                    )).item()
                    , 0)


if __name__ == '__main__':
    unittest.main()
