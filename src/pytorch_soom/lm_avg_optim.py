import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd.functional import hessian
from torch.func import functional_call
from .second_order_optimizer import SecondOrderOptimizer, fix_stability, pinv_svd_trunc
import warnings
from copy import copy, deepcopy


class LMAvg(SecondOrderOptimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
    and the matlab implementation of 'learnlm' https://es.mathworks.com/help/deeplearning/ref/trainlm.html#d126e69092


    """

    def __init__(
        self,
        params,
        model,
        lr=1,
        mu=1,
        mu_dec=0.1,
        mu_max=1e10,
        use_diagonal=True,
        hessian_approx=False,
        debug_stability=True,
        **kwargs,
    ):
        assert lr > 0, "Learning rate must be a positive number"

        super().__init__(params, {"lr": lr, "mu": mu})

        self.hessian_approx = hessian_approx

        self.mu = mu
        self.mu_dec = mu_dec
        self.mu_max = mu_max

        self._model = model
        self._params = self.param_groups[0]["params"]

        self._j_list = []
        self._h_list = []
        self.prev_loss = None
        self._prev_params = deepcopy(self.param_groups[0]["params"])
        self.use_diagonal = use_diagonal
        self.debug_stability = debug_stability

    def _apply_gradients(self, params, d_p_list, h_list, lr):
        """ """

        for i, param in enumerate(params):
            d_p = d_p_list[i]
            h = h_list[i]

            if h is None:
                param.add_(d_p, alpha=-lr)
                break

            diag_vec = h.diagonal() + torch.finfo(h.dtype).eps * 1
            h.as_strided([h.size(0)], [h.size(0) + 1]).copy_(diag_vec)

            if self.use_diagonal:
                adjustment = h.diagonal()
                h_adjusted = (1-self.mu) * h + self.mu * adjustment

                # Use truncated SVD pseudoinverse to address numerical instability
                h_i = pinv_svd_trunc(h_adjusted)
            else:
                adjustment = torch.eye(h.shape[0], device=h.device)
                h_adjusted = (1-self.mu) * h + self.mu * adjustment

                h_i = h_adjusted.pinverse()

            assert h_i.shape[-1] == d_p.flatten().shape[0], "Tensor dimension mismatch"

            d2_p = h_i.matmul(d_p.flatten()).reshape(d_p_list[i].shape)
            param.add_(d2_p, alpha=-lr)

    @torch.no_grad()
    def step(self, x, y, loss_fn, closure=None):
        """ """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        parameters = dict(self._model.named_parameters())
        keys, values = zip(*parameters.items())

        def func(*params):
            out = functional_call(self._model, {n: p for n, p in zip(keys, params)}, x)
            return loss_fn(out, y)

        self._h_list = []
        if not self.hessian_approx:
            self._h_list = torch.autograd.functional.hessian(func, values, create_graph=True)
            self._h_list = [self._h_list[i][i] for i, _ in enumerate(self._h_list)]

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            lr = group["lr"]

            if self.hessian_approx:
                self._j_list = torch.autograd.functional.jacobian(func, values, create_graph=False)
                for i, j in enumerate(self._j_list):
                    j = j.flatten()
                    h = torch.outer(j, j)
                    self._h_list.append(h)

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            h_list = [self._reshape_hessian(h) for h in self._h_list]

            self._apply_gradients(params_with_grad, d_p_list, h_list, lr)

        return loss

    def update(self, loss):
        loss_val = loss.detach().item()

        if self.prev_loss is None or loss_val < self.prev_loss:
            self.prev_loss = loss_val
            self._prev_params = deepcopy(self._params)
            self.mu = self.mu * self.mu_dec
        else:
            self._params = self._prev_params
            self.mu = 1 - ((1 - self.mu) * self.mu_dec)
