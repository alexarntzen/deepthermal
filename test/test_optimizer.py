import unittest
import torch
import torch.utils
import torch.utils.data
import numpy as np

from deepthermal.optimization import argmin


class TestOptimizer(unittest.TestCase):

    @staticmethod
    def get_unconstrained_cost(d=2):
        a = torch.rand(d)

        def cost(v):
            return torch.sum((v - a) ** 2)

        optim = a
        return cost, optim

    @staticmethod
    def get_unconstrained_cost_quad(d=2, a=0, b=1):
        A = torch.rand(d, d)
        optim = torch.rand(d) * (b - a) + a

        # symmetric diagonally dominant matrix is positive definite
        A = 0.5 * (A + A.T) + d * torch.eye(d)

        def cost(v):
            return (v - optim) @ A @ (v - optim)

        return cost, optim

    @staticmethod
    def get_constrained_cost():
        a = torch.rand(1)
        b = torch.rand(1)

        def cost(v):
            return (v[1] - a * v[0] - b) ** 2 + v[0]

        optim = torch.tensor([0, b])
        return cost, optim

    def test_unconstrained(self):
        print("\n\n Testing the optimizer on 20 unconstrained problems:")
        for d in range(1, 10):
            cost, optim = self.get_unconstrained_cost(d)
            x_0 = torch.rand(d)
            x_opt = argmin(cost, x_0)
            self.assertAlmostEqual(torch.max(torch.abs(x_opt - optim)).item(), 0, delta=1e-5)

        for d in range(1, 10):
            cost, optim = self.get_unconstrained_cost_quad(d, a=-10, b=10)
            x_0 = torch.rand(d)
            x_opt = argmin(cost, x_0)
            self.assertAlmostEqual(torch.max(torch.abs(x_opt - optim)).item(), 0, delta=1e-5)

    def test_irrelevant_constraint(self):
        print("\n\n Testing the optimizer on 20 unconstrained problems with the constrained solver:")
        # simple problem with solution in box
        for d in range(1, 10):
            cost, optim = self.get_unconstrained_cost(d)
            x_0 = torch.rand(d)
            x_opt = argmin(cost, x_0, box_constraint=[0, 1])
            self.assertAlmostEqual(torch.max(torch.abs(x_opt - optim)).item(), 0, delta=1e-5)

        # quad problem with solution in box
        for d in range(1, 10):
            cost, optim = self.get_unconstrained_cost_quad(d)
            x_0 = torch.rand(d)
            x_opt = argmin(cost, x_0, box_constraint=[0, 1])
            self.assertAlmostEqual(torch.max(torch.abs(x_opt - optim)).item(), 0, delta=1e-5)

    def test_constrained(self):
        print("\n\n Testing optimizer on a constrained problem:")
        cost, optim = self.get_constrained_cost()
        x_0 = torch.rand(2)
        x_opt = argmin(cost, x_0, box_constraint=[0, 1])
        self.assertAlmostEqual(torch.max(torch.abs(x_opt - optim)).item(), 0, delta=1e-5)


if __name__ == '__main__':
    unittest.main()
