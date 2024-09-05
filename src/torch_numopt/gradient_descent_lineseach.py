from __future__ import annotations
from typing import Iterable
import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd.functional import hessian
from torch.func import functional_call
from .line_search_mixin import LineSearchMixin
from .utils import fix_stability, pinv_svd_trunc


class GradientDescentLS(LineSearchMixin, Optimizer):
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

    def _apply_gradients(self, params, d_p_list, lr, eval_model):
        """ """

        step_dir = d_p_list

        if self.line_search_method == "backtrack":
            new_params = self.backtrack_wolfe(params, step_dir, d_p_list, lr, eval_model, self.c1, self.c2, self.tau, self.line_search_cond)
        elif self.line_search_method == "const":
            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))

        # Apply new parameters
        for param, new_param in zip(params, new_params):
            param.copy_(new_param)

    @torch.no_grad()
    def step(self, x, y, loss_fn, closure=None):
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

            self._apply_gradients(params=params_with_grad, d_p_list=d_p_list, lr=lr, eval_model=eval_model)
