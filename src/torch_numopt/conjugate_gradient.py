from __future__ import annotations
from typing import Iterable
import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd.functional import hessian
from torch.func import functional_call
from .utils import fix_stability, pinv_svd_trunc
from copy import copy


class ConjugateGradient(Optimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
    https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
    https://arxiv.org/abs/2201.08568

    Parameters
    ----------

    params: Iterable[nn.parameter.Parameter]
        Parameters of the model to be optimized.
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
        params: Iterable[nn.parameter.Parameter],
        lr: float,
        model: nn.Module,
        c1: float = 1e-4,
        c2: float = 0.9,
        tau: float = 0.1,
        line_search_method: str = "const",
        line_search_cond: str = "armijo",
        cg_method: str = "FR",
        **kwargs,
    ):
        assert lr > 0, "Learning rate must be a positive number."

        super().__init__(params, {"lr": lr})

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

    def _line_search_cond(self, params, new_params, step_dir, lr, loss, new_loss, grad):
        accepted = True

        dir_deriv = sum(
            [
                torch.dot(p_grad.flatten(), p_step.flatten())
                for p_grad, p_step in zip(grad, step_dir)
            ]
        )

        if self.line_search_cond == "armijo":
            accepted = new_loss <= loss + self.c1 * lr * dir_deriv
        elif self.line_search_cond == "wolfe":
            new_grad = torch.autograd.grad(new_loss, new_params)
            new_dir_deriv = sum(
                [
                    torch.dot(p_grad.flatten(), p_step.flatten())
                    for p_grad, p_step in zip(new_grad, step_dir)
                ]
            )
            armijo = new_loss <= loss + self.c1 * lr * dir_deriv
            curv_cond = new_dir_deriv >= self.c2 * dir_deriv
            accepted = armijo and curv_cond
        elif self.line_search_cond == "strong-wolfe":
            new_grad = torch.autograd.grad(new_loss, new_params)
            new_dir_deriv = sum(
                [
                    torch.dot(p_grad.flatten(), p_step.flatten())
                    for p_grad, p_step in zip(new_grad, step_dir)
                ]
            )
            armijo = new_loss <= loss + self.c1 * lr * dir_deriv
            curv_cond = abs(new_dir_deriv) <= self.c2 * abs(dir_deriv)
            accepted = armijo and curv_cond
        elif self.line_search_cond == "goldstein":
            accepted = (
                loss + (1 - self.c1) * lr * dir_deriv
                <= new_loss
                <= loss + self.c1 * lr * dir_deriv
            )
        else:
            raise ValueError(
                f"Line search condition {self.line_search_cond} does not exist."
            )

        return accepted

    @torch.enable_grad()
    def _backtrack_wolfe(self, params, step_dir, grad, lr_init, eval_model):
        lr = lr_init

        loss = eval_model(*params)

        new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
        new_loss = eval_model(*new_params)

        while not self._line_search_cond(
            params, new_params, step_dir, lr, loss, new_loss, grad
        ):
            lr *= self.tau

            # Evaluate model with new lr
            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)

            if lr <= 1e-10:
                break

        return new_params

    def _apply_gradients(self, params, d_p_list, lr, eval_model):
        """ """

        step_dir = self._get_step_directions(d_p_list)

        if self.line_search_method == "backtrack":
            new_params = self._backtrack_wolfe(
                params, step_dir, d_p_list, lr, eval_model
            )
        elif self.line_search_method == "const":
            with torch.enable_grad():
                new_params = tuple(
                    p - lr * p_step for p, p_step in zip(params, step_dir)
                )

        # Apply new parameters
        for param, new_param in zip(params, new_params):
            param.copy_(new_param)

    def _get_step_directions(self, d_p_list):
        """ """
        if self.prev_dir is None:
            return d_p_list

        next_grad = [None] * len(d_p_list)
        for idx, (res, prev_res) in enumerate(zip(d_p_list, self.prev_dir)):
            res = res.view((-1, 1))
            prev_res = prev_res.view((-1, 1))

            if self.cg_method == "FR":
                beta = (res.T @ res) / (prev_res.T @ prev_res)
            elif self.cg_method == "PR":
                beta = (res.T @ (res - prev_res)) / (prev_res.T @ prev_res)
            elif self.cg_method == "PRP+":
                beta = torch.relu((res.T @ (res - prev_res)) / (prev_res.T @ prev_res))

            res_reshaped = res.view(next_grad[idx].shape)
            next_grad[idx].add_(res_reshaped, alpha=-beta)

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
            out = functional_call(
                self._model, dict(zip(self._param_keys, input_params)), x
            )
            return loss_fn(out, y)

        def get_residuals(*input_params):
            out = functional_call(
                self._model, dict(zip(self._param_keys, input_params)), x
            )
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

            self._apply_gradients(
                params=params_with_grad, d_p_list=d_p_list, lr=lr, eval_model=eval_model
            )
