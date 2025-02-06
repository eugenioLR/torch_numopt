from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
from torch.func import functional_call
from .line_search_optimizer import LineSearchOptimizer
from .custom_optimizer import CustomOptimizer


class GradientDescentLS(LineSearchOptimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py

    Parameters
    ----------

    model: nn.Module
        The model to be optimized
    lr: float
        Maximum learning rate in backtracking line search, if the learning rate is set as constant, this will be the value used.
    c1: float
        Coefficient of the sufficient increase condition in backtracking line search.
    c2: float
        Coefficient used in the second condition for wolfe conditions.
    tau: float
        Factor used to reduce the step size in each step of the backtracking line search.
    line_search_method: str
        Method used for line search, options are "backtrack" and "constant".
    line_search_cond: str
        Condition to be used in backtracking line search, options are "armijo", "wolfe", "strong-wolfe" and "goldstein".
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        line_search_method: str = "const",
        line_search_cond: str = "armijo",
        **kwargs,
    ):
        assert lr > 0, "Learning rate must be a positive number."

        super().__init__(model.parameters(), {"lr": lr})

        self._model = model
        self._param_keys = dict(model.named_parameters()).keys()
        self._params = self.param_groups[0]["params"]

        self.c1 = c1
        self.c2 = c2
        self.tau = tau
        self.line_search_method = line_search_method
        self.line_search_cond = line_search_cond

    def get_step_direction(self, d_p_list, h_list):
        return d_p_list

    @torch.no_grad()
    def step(self, x, y, loss_fn, closure=None):
        if closure is not None:
            raise NotImplementedError("This optimizer cannot handle closures.")

        model_params = tuple(self._model.parameters())

        def eval_model(*input_params):
            out = functional_call(self._model, dict(zip(self._param_keys, input_params)), x)
            return loss_fn(out, y)

        for group in self.param_groups:
            lr = group["lr"]

            # Calculate gradients
            params_with_grad = []
            d_p_list = []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            self.apply_gradients(params=params_with_grad, d_p_list=d_p_list, lr=lr, eval_model=eval_model)
