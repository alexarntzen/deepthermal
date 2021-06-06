import unittest
import torch
import torch.utils
import torch.utils.data

import numpy as np

from deepthermal.forcasting import TimeSeriesDataset, get_structured_prediction


class TestPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_samples = 1000
        cls.x = 2 * np.pi * torch.rand((cls.n_samples, 1))

    def test_dataset_params(self):
        print("\n\n Test whether TimeSeriesDatasets stores the right parameters:")
        for label_width in range(2, 20):
            dataset = TimeSeriesDataset(
                self.x, input_width=50, label_width=label_width, offset=20
            )
            w_len = self.n_samples - dataset.input_width - dataset.offset + 1
            w_max_len = self.n_samples - dataset.input_width + 1
            self.assertEqual(
                dataset[w_len - 1][0].shape, dataset[w_max_len - 1][0].shape
            )
            self.assertEqual(
                dataset[w_len - 1][1].shape, dataset[w_max_len - 1][1].shape
            )
            with self.assertRaises(IndexError):
                dataset[w_max_len]

            dataset = TimeSeriesDataset(
                self.x[:, 0], input_width=10, label_width=label_width, offset=20
            )
            w_len = self.n_samples - dataset.input_width - dataset.offset + 1
            w_max_len = self.n_samples - dataset.input_width + 1
            self.assertEqual(
                dataset[w_len - 1][0].shape, dataset[w_max_len - 1][0].shape
            )
            self.assertEqual(
                dataset[w_len - 1][1].shape, dataset[w_max_len - 1][1].shape
            )
            with self.assertRaises(IndexError):
                dataset[w_max_len]

            # make sure the last index is in the last dataset
            # and the same for first index
            self.assertEqual(dataset[dataset.max_len - 1][0][-1], self.x[-1, 0])
            self.assertEqual(dataset[len(dataset) - 1][1][-1], self.x[-1, 0])
            self.assertEqual(dataset[0][0][0], self.x[0, 0])

    def test_structured_prediction(self):
        print("\n\n Test whether prediction function uses last time index:")
        for label_width in range(1, 20):
            for offset in range(0, 20):
                dataset = TimeSeriesDataset(
                    self.x, input_width=50, label_width=label_width, offset=offset
                )
                t_indices, y_pred = get_structured_prediction(torch.sin, dataset)
                last_pred_index = self.x.shape[0] + offset - 1
                self.assertTrue(max(t_indices) == last_pred_index)


if __name__ == "__main__":
    unittest.main()
