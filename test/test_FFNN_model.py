import unittest
import torch

import torch.utils
import torch.utils.data

import numpy as np

from deepthermal.FFNN_model import get_trained_model, FFNN, fit_FFNN, init_xavier
from deepthermal.validation import (
    create_subdictionary_iterator,
    get_RRSE,
    k_fold_cv_grid,
)


class TestOnSimpleFunctionApprox(unittest.TestCase):
    @staticmethod
    def exact_solution(x):
        return torch.sin(x)

    def setUp(self):
        n_samples = 1000
        sigma = 0.0

        self.x = 2 * np.pi * torch.rand((n_samples, 1))
        self.y = self.exact_solution(self.x) * (1 + sigma * torch.randn(self.x.shape))
        self.x_test = torch.linspace(0, 2 * np.pi, 10000).reshape(-1, 1)
        self.y_test = self.exact_solution(self.x_test)
        self.data = torch.utils.data.TensorDataset(self.x, self.y)
        self.data_test = torch.utils.data.TensorDataset(self.x_test, self.y_test)

    def test_approx_error(self):
        print("\n\n Approximating the sine function:")

        model_params = {
            "model": FFNN,
            "input_dimension": 1,
            "output_dimension": 1,
            "n_hidden_layers": 5,
            "neurons": 10,
            "activation": "relu",
        }
        training_params = {
            "num_epochs": 500,
            "batch_size": 500,
            "regularization_exp": 2,
            "regularization_param": 1e-6,
            "optimizer": "ADAM",
            "learning_rate": 0.01,
            "init": init_xavier,
            "init_weight_seed": 20,
        }

        model, loss_history_train, loss_history_val = get_trained_model(
            model_params, training_params, data=self.data
        )

        rel_test_error = get_RRSE(model, self.data_test)
        self.assertAlmostEqual(0, rel_test_error, delta=0.1)

    def test_k_fold_cv_grid(self):
        print("\n\n Approximating the sine function with cross validation grid:")

        model_params = {
            "model": [FFNN],
            "input_dimension": [1],
            "output_dimension": [1],
            "n_hidden_layers": [5],
            "neurons": [10, 20],
            "activation": [
                "relu",
            ],
        }
        training_params = {
            "num_epochs": [500],
            "batch_size": [500],
            "regularization_exp": [2],
            "regularization_param": [1e-6],
            "optimizer": ["ADAM"],
            "learning_rate": [0.01],
            "init_weight_seed": [15],
            "init": [init_xavier],
        }

        model_params_iterator = create_subdictionary_iterator(model_params)
        training_params_iterator = create_subdictionary_iterator(training_params)

        cv_results = k_fold_cv_grid(
            model_params_iterator,
            training_params_iterator,
            data=self.data,
            fit=fit_FFNN,
            get_error=get_RRSE,
            folds=3,
        )
        avg_rel_val_errors = torch.mean(
            torch.tensor(cv_results["rel_val_errors"]), dim=1
        )
        self.assertAlmostEqual(0, torch.max(avg_rel_val_errors).item(), delta=0.1)

        for submodels in cv_results["models"]:
            for model in submodels:
                rel_test_error = get_RRSE(model, self.data_test)
                self.assertAlmostEqual(0, rel_test_error, delta=0.1)

    def test_k_fold_cv_grid_partial(self):
        print(
            "\n\n Approximating the sine function with partial cross validation grid:"
        )
        model_params = {
            "model": [FFNN],
            "input_dimension": [1],
            "output_dimension": [1],
            "n_hidden_layers": [5],
            "neurons": [10, 20],
            "activation": ["relu"],
        }
        training_params = {
            "num_epochs": [500],
            "batch_size": [500],
            "regularization_exp": [2],
            "regularization_param": [1e-6],
            "optimizer": ["ADAM"],
            "learning_rate": [0.01],
            "init_weight_seed": [25],
            "init": [init_xavier],
        }

        model_params_iterator = create_subdictionary_iterator(model_params)
        training_params_iterator = create_subdictionary_iterator(training_params)

        cv_results = k_fold_cv_grid(
            model_params=model_params_iterator,
            fit=fit_FFNN,
            training_params=training_params_iterator,
            data=self.data,
            folds=5,
            partial=True,
            get_error=get_RRSE,
        )

        avg_rel_val_errors = torch.mean(
            torch.tensor(cv_results["rel_val_errors"]), dim=1
        )
        self.assertAlmostEqual(0, torch.max(avg_rel_val_errors).item(), delta=0.1)

        for submodels in cv_results["models"]:
            for model in submodels:
                rel_test_error = get_RRSE(model, self.data_test)
                self.assertAlmostEqual(0, rel_test_error, delta=0.1)


if __name__ == "__main__":
    unittest.main()
