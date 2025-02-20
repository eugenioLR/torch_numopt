from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn
from torch.autograd.functional import hessian
from torch.func import functional_call
from .second_order_optimizer import SecondOrderOptimizer
from .utils import fix_stability, pinv_svd_trunc
import warnings
from copy import deepcopy, copy


class LM(SecondOrderOptimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
    and the matlab implementation of 'learnlm' https://es.mathworks.com/help/deeplearning/ref/trainlm.html#d126e69092

    Parameters
    ----------

    model: nn.Module
        The model to be optimized
    lr: float
        Maximum learning rate in backtracking line search, if the learning rate is set as constant, this will be the value used.
    mu: float
        Initial value for the coefficient used when adding a diagonal matrix to the Hessian approximation.
    mu_dec: float
        Factor with which to decrease the coefficient of the diagonal matrix if the previous iteration didn't improve the model.
    mu_max: float
        Factor with which to increase the coefficient of the diagonal matrix if the previous iteration improved the model.
    use_diagonal: bool
        Whether to use the diagonal of the Hessian approximation instead of an identity matrix to adjust the Hessian matrix.
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
        mu: float = 1,
        mu_dec: float = 0.1,
        mu_max: float = 1e10,
        fletcher: bool = False,
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

        self.mu = mu
        self.mu_dec = mu_dec
        self.mu_max = mu_max
        self.fletcher = fletcher

        self.prev_loss = None

        # Coefficients for the strong-wolfe conditions
        self.c1 = c1
        self.c2 = c2
        self.tau = tau
        self.line_search_method = line_search_method
        self.line_search_cond = line_search_cond

        self.solver = solver

        if fletcher and solver == "solve":
            warnings.warn("Using 'solve' with fletcher's method usually doesn't work very well. Try using 'pinv' instead.")

    def get_step_direction(self, d_p_list, h_list):
        dir_list = [None] * len(d_p_list)
        for i, (d_p, h) in enumerate(zip(d_p_list, h_list)):
            if self.fletcher:
                h_adjusted = h + self.mu * h.diagonal()
            else:
                h_adjusted = h + self.mu * torch.eye(h.shape[0], device=h.device)

            match self.solver:
                case "pinv":
                    if self.fletcher:
                        h_i = pinv_svd_trunc(h_adjusted)
                    else:
                        h_i = h_adjusted.pinverse()
                    
                    d2_p = (h_i @ d_p.flatten()).reshape(d_p.shape)
                case "solve":
                    d2_p = torch.linalg.solve(h_adjusted, d_p.flatten()).reshape(d_p.shape)

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
                approx_h = j.T @ j
                h_list[j_idx] = self._reshape_hessian(approx_h)

            # Calculte gradients
            params_with_grad = []
            d_p_list = []
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            self.apply_gradients(params=params_with_grad, d_p_list=d_p_list, h_list=h_list, lr=lr, eval_model=eval_model)

    def update(self, loss):
        loss_val = loss.detach().item()

        if self.prev_loss is None:
            self.prev_loss = loss_val
            self._prev_params = deepcopy(self._params)
        elif loss_val <= self.prev_loss:
            self.prev_loss = loss_val
            self._prev_params = deepcopy(self._params)
            self.mu *= self.mu_dec
        else:
            self._params = self._prev_params
            self.mu /= self.mu_dec

        if self.mu >= self.mu_max:
            self.mu = self.mu_max
