import unittest
import torch
import numpy as np

from deepthermal.multilevel import (
    MultilevelDataset,
    get_level_dataset,
    fit_multilevel_FFNN,
    MultilevelFFNN,
    get_multilevel_RRSE,
)
from deepthermal.FFNN_model import get_trained_model, init_xavier


class TestMultilevel(unittest.TestCase):
    def setUp(self):
        self.n_samples = 2000
        self.x = 2 * np.pi * torch.rand((self.n_samples, 1))
        self.y_0 = torch.sin(self.x)
        self.y_1 = self.y_0[: self.n_samples // 2]
        self.y_2 = self.y_0[: self.n_samples // 4]
        self.datalist = [
            self.y_0.detach().clone(),
            self.y_1.detach().clone(),
            self.y_2.detach().clone(),
        ]
        self.len_list = [self.n_samples, self.n_samples // 2, self.n_samples // 4]

    def test_dataset_params(self):
        print(
            "\n\n Test whether MultilevelDataset "
            "stores the right data and return the right dataset:"
        )
        dataset = MultilevelDataset(self.x, self.datalist)

        x_data, y_data = dataset[:]
        for level in range(len(y_data)):
            self.assertEqual(x_data.size(0), y_data[level].size(0))
            self.assertEqual(
                torch.sum(torch.isnan(y_data[level])),
                self.n_samples - self.len_list[level],
            )
            x_train, y_train = get_level_dataset(x_data, y_data, level)
            self.assertEqual(x_train.size(0), y_train.size(0))
            self.assertEqual(
                torch.max(torch.abs(self.x[: self.len_list[level]] - x_train)).item(), 0
            )
            if level == 0:
                self.assertEqual(torch.max(torch.abs(y_train - self.y_0)).item(), 0)
            elif level > 0:
                self.assertEqual(
                    torch.max(
                        torch.abs(
                            y_train
                            - self.datalist[level]
                            + self.datalist[level - 1][: self.len_list[level]]
                        )
                    ).item(),
                    0,
                )

    def test_multilevel_approx_error(self):
        print("\n\n Approximating the sine function with a multilevel model:")
        ml_dataset = MultilevelDataset(self.x, self.datalist)
        model_params = {
            "model": MultilevelFFNN,
            "input_dimension": 1,
            "output_dimension": 1,
            "n_hidden_layers": 5,
            "neurons": 10,
            "activation": "relu",
            "levels": 3,
        }
        training_params = {
            "num_epochs": 500,
            "batch_size": 500,
            "regularization_exp": 2,
            "regularization_param": 1e-6,
            "optimizer": "ADAM",
            "learning_rate": 0.01,
            "init_weight_seed": 20,
            "init": init_xavier,
        }
        model, loss_history_train, loss_history_val = get_trained_model(
            model_param=model_params,
            training_param=training_params,
            data=ml_dataset,
            fit=fit_multilevel_FFNN,
        )

        rel_test_error = get_multilevel_RRSE(model, ml_dataset)
        self.assertAlmostEqual(0, rel_test_error, delta=0.1)


if __name__ == "__main__":
    unittest.main()
