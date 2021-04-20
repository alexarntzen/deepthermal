import unittest
import torch
import torch.utils
import torch.utils.data
import pandas as pd
import numpy as np

from deepthermal.FFNN_model import init_xavier, FFNN, fit_FFNN
from deepthermal.validation import create_subdictionary_iterator, k_fold_CV_grid
from deepthermal.task1_submission import FINAL_MODEL_PARAMS, FINAL_TRAINING_PARAMS


class TestTask1Model(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        PATH_TRAINING_DATA = "Task1/TrainingData.txt"
        PATH_TESTING_POINTS = "Task1/TestingData.txt"
        # Data frame with data
        df_train = pd.read_csv(PATH_TRAINING_DATA, dtype=np.float32)
        df_test = pd.read_csv(PATH_TESTING_POINTS, dtype=np.float32)

        # Load data
        x_train_ = torch.tensor(df_train[["t"]].values)
        cls.y_train = torch.tensor(df_train[["tf0", "ts0"]].values)
        x_test_ = torch.tensor(df_test[["t"]].values)

        # Normalize values
        X_TRAIN_MEAN = torch.mean(x_train_)
        X_TRAIN_STD = torch.std(x_train_)
        cls.x_train = (x_train_ - X_TRAIN_MEAN) / X_TRAIN_STD
        cls.x_test = (x_test_ - X_TRAIN_MEAN) / X_TRAIN_STD

    def test_task1_model(self):
        model_params_iter = create_subdictionary_iterator(FINAL_MODEL_PARAMS)
        training_params_iter = create_subdictionary_iterator(FINAL_TRAINING_PARAMS)
        models, rel_training_error, rel_val_errors = k_fold_CV_grid(Model=FFNN, model_param_iter=model_params_iter,
                                                                    fit=fit_FFNN,
                                                                    training_param_iter=training_params_iter,
                                                                    x=self.x_train, y=self.y_train, init=init_xavier,
                                                                    partial=False,
                                                                    verbose=True, k=5)

        avg_rel_val_error = sum(rel_val_errors[0]) / len(rel_val_errors[0])
        print(f"Average relative validation error: {avg_rel_val_error * 100}%")
        self.assertAlmostEqual(0, avg_rel_val_error, delta=0.01)


if __name__ == '__main__':
    unittest.main()
