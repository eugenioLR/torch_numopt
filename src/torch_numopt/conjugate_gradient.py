from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
from torch.func import functional_call
from .line_search_mixin import LineSearchMixin
from .custom_optimizer import CustomOptimizer
from copy import copy


class ConjugateGradient(LineSearchMixin, CustomOptimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
    https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
    https://arxiv.org/abs/2201.08568

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
    cg_method: str
        Formula used to calculate the conjugate gradient, options are "FR", "PR" and "PRP+".
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
        cg_method: str = "FR",
        **kwargs,
    ):
        assert lr > 0, "Learning rate must be a positive number."

        super().__init__(model.parameters(), {"lr": lr})

        self._model = model
        self._param_keys = dict(model.named_parameters()).keys()
        self._params = self.param_groups[0]["params"]

        # Conjugate gradient memory
        self.prev_residual = None
        self.prev_dir = None
        self.cg_method = cg_method

        # Coefficients for the strong-wolfe conditions
        self.c1 = c1
        self.c2 = c2
        self.tau = tau
        self.line_search_method = line_search_method
        self.line_search_cond = line_search_cond

    def _apply_gradients(self, params, d_p_list, lr, eval_model):
        """ """

        step_dir = self._get_step_directions(d_p_list)

        if self.line_search_method == "backtrack":
            new_params = self.backtrack_wolfe(params, step_dir, d_p_list, lr, eval_model, self.c1, self.c2, self.tau, self.line_search_cond)
        elif self.line_search_method == "const":
            with torch.enable_grad():
                new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))

        # Apply new parameters
        for param, new_param in zip(params, new_params):
            param.copy_(new_param)

    def _get_step_directions(self, d_p_list):
        """ """

        if self.prev_dir is None:
            self.prev_residual = self.prev_dir = d_p_list
            return d_p_list

        # next_grad = [None] * len(d_p_list)
        next_grad = deepcopy(d_p_list)
        for idx, (res, prev_res, prev_dir) in enumerate(zip(d_p_list, self.prev_residual, self.prev_dir)):
            res = res.view((-1, 1))
            prev_res = prev_res.view((-1, 1))
            prev_dir = prev_dir.view((-1, 1))

            if self.cg_method == "FR":
                beta = (res.T @ res) / (prev_res.T @ prev_res)
            elif self.cg_method == "PR":
                beta = (res.T @ (res - prev_res)) / (prev_res.T @ prev_res)
            elif self.cg_method == "PRP+":
                beta = torch.relu((res.T @ (res - prev_res)) / (prev_res.T @ prev_res))
            elif self.cg_method == "HS":
                beta =  (res.T @ (res - prev_res)) / (-prev_dir.T @ (res - prev_res))
            elif self.cg_method == "DY":
                beta =  (res.T @ res) / (-prev_dir.T @ (res - prev_res))
            
            # beta = torch.clamp(beta, min=-1e6, max=1e6)
            if not torch.isfinite(beta):
                beta = torch.zeros(1)

            res_reshaped = res.view(next_grad[idx].shape)
            next_grad[idx].add_(res_reshaped, alpha=-beta.item())
            if not torch.any(torch.isfinite(next_grad[idx])):
                print(beta)

        self.prev_residual = d_p_list
        self.prev_dir = next_grad

        return next_grad

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

            # Calculate gradients
            params_with_grad = []
            d_p_list = []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            self._apply_gradients(params=params_with_grad, d_p_list=d_p_list, lr=lr, eval_model=eval_model)
