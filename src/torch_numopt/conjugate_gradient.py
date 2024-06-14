import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd.functional import hessian
from torch.func import functional_call
from .utils import fix_stability, pinv_svd_trunc


class ConjugateGradient(Optimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
    https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf           

    """

    def __init__(
        self,
        params,
        lr,
        model,
        c1=1e-4,
        c2=0.9,
        tau=0.1,
        line_search_method="const",
        line_search_cond="armijo",
        cg_method="FR",
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

    def _apply_gradients(self, params, d_p_list, lr, eval_model):
        """ """

        step_dir = self._get_step_directions(d_p_list)

        if self.line_search_method == "backtrack":
            new_params = self._backtrack_wolfe(params, step_dir, d_p_list, lr, eval_model)
        elif self.line_search_method == "const":
            with torch.enable_grad():
                new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
        
        self._compute_next_dir(new_params, eval_model)

        # Apply new parameters
        for param, new_param in zip(params, new_params):
            param.copy_(new_param)

    def _get_step_directions(self, d_p_list):
        """ """
        dir_list = self.prev_dir if self.prev_dir is not None else d_p_list
        return dir_list
    
    def _compute_next_dir(self, new_params, eval_model):
        """ """
        with torch.enable_grad():
            new_loss = eval_model(*new_params)
            new_grad = torch.autograd.grad(new_loss, new_params)

        if self.prev_dir is None:
            self.prev_residuals = new_grad
            self.prev_dir = new_grad
            return

        for idx, (res, prev_res) in enumerate(zip(new_grad, self.prev_residuals)):
            res = res.view((-1, 1))
            prev_res = prev_res.view((-1, 1))

            if self.cg_method == "FR":
                beta = (res.T @ res) / (prev_res.T @ prev_res)
            elif self.cg_method == "PR":
                beta = torch.relu((res.T @ (res - prev_res)) / (prev_res.T @ prev_res))
            beta = beta.item()

            self.prev_dir[idx].add_(res.view(self.prev_dir[idx].shape), alpha=-beta)
        
        self.prev_residuals = new_grad

    @torch.no_grad()
    def step(self, x, y, loss_fn, closure=None):
        """ """

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


