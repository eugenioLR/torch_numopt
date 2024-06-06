import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
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
    """

    def __init__(
        self,
        params,
        model,
        lr,
        mu=1,
        mu_dec=0.1,
        mu_max=1e10,
        use_diagonal=True,
        c1=1e-4,
        c2=0.9,
        tau=0.1,
        line_search_method="const",
        line_search_cond="armijo",
        **kwargs,
    ):
        assert lr > 0, "Learning rate must be a positive number."

        super().__init__(params, {"lr": lr})

        self._model = model
        self._param_keys = dict(model.named_parameters()).keys()
        self._params = self.param_groups[0]["params"]

        self.mu = mu
        self.mu_dec = mu_dec
        self.mu_max = mu_max
        self.use_diagonal = use_diagonal

        self.prev_loss = None

        # Coefficients for the strong-wolfe conditions
        self.c1 = c1
        self.c2 = c2
        self.tau = tau
        self.line_search_method = line_search_method
        self.line_search_cond = line_search_cond

    def _line_search_cond(self, params, new_params, step_dir, lr, loss, new_loss, grad):
        accepted = True

        dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(grad, step_dir)])

        if self.line_search_cond == "armijo":
            accepted = new_loss <= loss + self.c1 * lr * dir_deriv
        elif self.line_search_cond == "wolfe":
            new_grad = torch.autograd.grad(new_loss, new_params)
            new_dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir)])
            armijo = new_loss <= loss + self.c1 * lr * dir_deriv
            curv_cond = new_dir_deriv >= self.c2 * dir_deriv
            accepted = armijo and curv_cond
        elif self.line_search_cond == "strong-wolfe":
            new_grad = torch.autograd.grad(new_loss, new_params)
            new_dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir)])
            armijo = new_loss <= loss + self.c1 * lr * dir_deriv
            curv_cond = abs(new_dir_deriv) <= self.c2 * abs(dir_deriv)
            accepted = armijo and curv_cond
        elif self.line_search_cond == "goldstein":
            accepted = loss + (1 - self.c1) * lr * dir_deriv <= new_loss <= loss + self.c1 * lr * dir_deriv
        else:
            raise ValueError(f"Line search condition {self.line_search_cond} does not exist.")

        return accepted

    @torch.enable_grad()
    def _backtrack_wolfe(self, params, step_dir, grad, lr_init, eval_model):
        lr = lr_init

        loss = eval_model(*params)

        new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
        new_loss = eval_model(*new_params)

        while not self._line_search_cond(params, new_params, step_dir, lr, loss, new_loss, grad):
            lr *= self.tau

            # Evaluate model with new lr
            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)

            if lr <= 1e-10:
                break

        return new_params

    def _apply_gradients(self, params, d_p_list, h_list, lr, eval_model):
        """ """

        step_dir = self._get_step_directions(d_p_list, h_list, lr)

        if self.line_search_method == "backtrack":
            new_params = self._backtrack_wolfe(params, step_dir, d_p_list, lr, eval_model)
        elif self.line_search_method == "const":
            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))

        # Apply new parameters
        for param, new_param in zip(params, new_params):
            param.copy_(new_param)

    def _get_step_directions(self, d_p_list, h_list, lr):
        dir_list = [None] * len(d_p_list)
        for i, (d_p, h) in enumerate(zip(d_p_list, h_list)):
            if self.use_diagonal:
                h_adjusted = h + self.mu * h.diagonal()

                # Use truncated SVD pseudoinverse to address numerical instability
                h_i = pinv_svd_trunc(h_adjusted)
            else:
                h_adjusted = h + self.mu * torch.eye(h.shape[0], device=h.device)

                h_i = h_adjusted.pinverse()

            d2_p = (h_i @ d_p.flatten()).reshape(d_p.shape)

            dir_list[i] = d2_p

        return dir_list

    @torch.no_grad()
    def step(self, x, y, loss_fn, closure=None):
        """ """

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

            self._apply_gradients(params=params_with_grad, d_p_list=d_p_list, h_list=h_list, lr=lr, eval_model=eval_model)

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
