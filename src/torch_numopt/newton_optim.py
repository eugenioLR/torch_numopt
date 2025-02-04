from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
from torch.autograd.functional import hessian
from torch.func import functional_call
from .second_order_optimizer import SecondOrderOptimizer
from .utils import fix_stability, pinv_svd_trunc
from .line_search_mixin import LineSearchMixin


class NewtonRaphson(LineSearchMixin, SecondOrderOptimizer):
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

        # Coefficients for the strong-wolfe conditions
        self.c1 = c1
        self.c2 = c2
        self.tau = tau
        self.line_search_method = line_search_method
        self.line_search_cond = line_search_cond

    def _apply_gradients(self, params, d_p_list, h_list, lr, eval_model):
        """ """

        step_dir = self._get_step_directions(d_p_list, h_list)

        if self.line_search_method == "backtrack":
            new_params = self.backtrack_wolfe(params, step_dir, d_p_list, lr, eval_model, self.c1, self.c2, self.tau, self.line_search_cond)
        elif self.line_search_method == "interpolate":
            # new_params = self.interpolate_quadratic(params, step_dir, d_p_list, lr, eval_model, self.c1, self.c2, self.line_search_cond)
            new_params = self.interpolate_cubic(params, step_dir, d_p_list, lr, eval_model, self.c1, self.c2, self.line_search_cond)
        elif self.line_search_method == "const":
            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))

        # Apply new parameters
        for param, new_param in zip(params, new_params):
            param.copy_(new_param)

    def _get_step_directions(self, d_p_list, h_list):
        dir_list = []
        for d_p, h in zip(d_p_list, h_list):
            # Handle issues with numerical stability
            h = fix_stability(h)
            h_i = h.pinverse()

            d2_p = (h_i @ d_p.flatten()).reshape(d_p.shape)
            dir_list.append(d2_p)

        return dir_list

    @torch.no_grad()
    def step(self, x, y, loss_fn, closure=None):
        if closure is not None:
            raise NotImplementedError("This optimizer cannot handle closures.")

        model_params = tuple(self._model.parameters())

        def eval_model(*input_params):
            out = functional_call(self._model, dict(zip(self._param_keys, input_params)), x)
            return loss_fn(out, y)

        # Calculate exact Hessian matrix
        h_list = torch.autograd.functional.hessian(eval_model, model_params, create_graph=True, vectorize=True)
        h_list = [self._reshape_hessian(h_list[i][i]) for i, _ in enumerate(h_list)]

        for group in self.param_groups:
            lr = group["lr"]

            # Calculate gradients
            params_with_grad = []
            d_p_list = []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            self._apply_gradients(params=params_with_grad, d_p_list=d_p_list, h_list=h_list, lr=lr, eval_model=eval_model)
