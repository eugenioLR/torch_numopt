from __future__ import annotations
from abc import ABC
import torch
from torch.optim.optimizer import Optimizer
from functools import reduce


class SecondOrderOptimizer(Optimizer, ABC):
    """
    Class for Optimization methods using second derivative information.
    """

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: callable,
        closure: callable = None,
    ):
        """
        Method to update the parameters of the Neural Network.

        x: torch.Tensor
            Inputs of the Neural Network.
        y: torch.Tensor
            Targets of the Neural Network.
        loss_fn: callable
            Loss function to be optimized.
        closure: callable
            Kept for compatibility, unused.
        """

    def update(self, loss):
        """
        Function to update the internal parameters of the optimization procedure.

        loss: float
            Loss of the Neural Network with the new parameters.
        """

    @staticmethod
    def _reshape_hessian(hess):
        """
        Procedure to reshape a misshapen hessian matrix.

        hess: torch.Tensor
            Misshapen hessian matrix.
        """

        if len(hess.shape) == 2:
            return hess

        if len(hess.shape) % 2 != 0:
            raise ValueError("Hessian has a weird ass shape.")

        # Divide shape in two halves, multiply each half to get total size
        new_shape = (
            reduce(lambda x, y: x * y, hess.size()[hess.dim() // 2 :]),
            reduce(lambda x, y: x * y, hess.size()[: hess.dim() // 2]),
        )

        assert (
            new_shape[0] == new_shape[1]
        ), "Something weird happened with the hessian size"

        return hess.reshape(new_shape)
