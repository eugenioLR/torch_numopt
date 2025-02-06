from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
from torch.autograd.functional import hessian
from torch.func import functional_call
from .second_order_optimizer import SecondOrderOptimizer
from .utils import fix_stability, pinv_svd_trunc
from copy import copy


class GaussNewton(SecondOrderOptimizer):
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
        solver: str = "solve",
        **kwargs,
    ):
        assert lr > 0, "Learning rate must be a positive number."

        super().__init__(model.parameters(), {"lr": lr})

        self._model = model
        self._param_keys = dict(model.named_parameters()).keys()
        self._params = self.param_groups[0]["params"]

        # Coefficients for the strong-wolfe conditions
        self.c1 = c1
        self.c2 = c2
        self.tau = tau
        self.line_search_method = line_search_method
        self.line_search_cond = line_search_cond

        self.solver = solver

    def get_step_direction(self, d_p_list, h_list):
        dir_list = [None] * len(d_p_list)
        for i, (d_p, h) in enumerate(zip(d_p_list, h_list)):
            if torch.linalg.cond(h) > 1e8:
                h = fix_stability(h)

            match self.solver:
                case "pinv":
                    h_i = h.pinverse()
                    d2_p = (h_i @ d_p.flatten()).reshape(d_p.shape)
                case "solve":
                    d2_p = torch.linalg.solve(h, d_p.flatten()).reshape(d_p.shape)

            dir_list[i] = d2_p

        return dir_list

    @torch.no_grad()
    def step(self, x, y, loss_fn, closure=None):
        if closure is not None:
            raise NotImplementedError("This optimizer cannot handle closures.")

        residual_fn = copy(loss_fn)
        residual_fn.reduction = "none"

        model_params = tuple(self._model.parameters())

        def eval_model(*input_params):
            out = functional_call(self._model, dict(zip(self._param_keys, input_params)), x)
            return loss_fn(out, y)

        def get_residuals(*input_params):
            out = functional_call(self._model, dict(zip(self._param_keys, input_params)), x)
            return residual_fn(out, y)

        for group in self.param_groups:
            lr = group["lr"]

            # Calculate approximate Hessian matrix
            j_list = torch.autograd.functional.jacobian(get_residuals, model_params, create_graph=False, vectorize=True)
            h_list = [None] * len(j_list)
            for j_idx, j in enumerate(j_list):
                j = j.flatten(start_dim=1)
                h_list[j_idx] = self._reshape_hessian(j.T @ j)

            # Calculte gradients
            params_with_grad = []
            d_p_list = []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            self.apply_gradients(params=params_with_grad, d_p_list=d_p_list, h_list=h_list, lr=lr, eval_model=eval_model)
