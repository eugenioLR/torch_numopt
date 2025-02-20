from __future__ import annotations
from abc import ABC
import torch
from functools import reduce
from .line_search_optimizer import LineSearchOptimizer


class SecondOrderOptimizer(LineSearchOptimizer, ABC):
    """
    Class for Optimization methods using second derivative information.
    """

    @staticmethod
    def _reshape_hessian(hess: torch.Tensor):
        """
        Procedure to reshape a misshapen hessian matrix.

        Parameters
        ----------

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

        assert new_shape[0] == new_shape[1], "Something weird happened with the hessian size"

        return hess.reshape(new_shape)
