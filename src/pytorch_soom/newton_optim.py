import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd.functional import hessian
from torch.func import functional_call
from .second_order_optimizer import SecondOrderOptimizer, fix_stability, pinv_svd_trunc
from copy import deepcopy


class Newton(SecondOrderOptimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
    """

    def __init__(
        self,
        params,
        lr,
        model,
        hessian_approx=False,
        c1=1e-4,
        c2=0.9,
        tau=0.1,
        line_search_method='const',
        line_search_cond='armijo',
        **kwargs,
    ):
        assert lr > 0, "Learning rate must be a positive number."

        super().__init__(params, {"lr": lr})

        self.hessian_approx = hessian_approx

        self._model = model
        self._params = self.param_groups[0]["params"]
        self._j_list = []
        self._h_list = []

        # Coefficients for the strong-wolfe conditions
        self.c1 = c1
        self.c2 = c2
        self.tau = tau
        self.line_search_method = line_search_method

    def _line_search_cond(self):
        accepted = True

        if self.line_search_method == 'armijo':
            pass
        elif self.line_search_method == 'wolfe':
            pass
        elif self.line_search_method == 'strong-wolfe':
            pass
        elif self.line_search_method == 'goldstein':
            pass
        
        return accepted


    @torch.enable_grad()
    def _backtrack_wolfe(self, params, step_dir, grad, lr_init, eval_model):
        lr = lr_init

        prev_loss = eval_model(*params)
        dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(grad, step_dir)])

        new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
        new_loss = eval_model(*new_params)
        new_grad = torch.autograd.grad(new_loss, new_params)

        armijo_cond = new_loss <= prev_loss + self.c1 * lr * dir_deriv
        while not armijo_cond:
            lr *= self.tau

            # Evaluate model with new lr
            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)
            new_grad = torch.autograd.grad(new_loss, new_params)

            # Recalculate conditions for new lr
            armijo_cond = new_loss <= prev_loss + self.c1 * lr * dir_deriv

        return new_params
    
    def _apply_gradients(self, params, d_p_list, h_list, lr, eval_model):
        """ """
        # _, params_tup = zip(*params.items())
        # params_list = [torch.Tensor(p) for p in params]

        step_dir = self._get_step_directions(d_p_list, h_list, lr)

        if self.line_search_method == "backtrack":
            new_params = self._backtrack_wolfe(params, step_dir, d_p_list, lr, eval_model)
        elif self.line_search_method == "const":
            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))

        # Apply new parameters
        for param, new_param in zip(params, new_params):
            param.copy_(new_param)

    def _get_step_directions(self, d_p_list, h_list, lr):
        dir_list = []
        for d_p, h in zip(d_p_list, h_list):
            # Handle issues with numerical stability
            h = fix_stability(h)
            h_i = h.pinverse()

            # h_i = pinv_svd_trunc(h, thresh=1e-5)

            # h_i = h.pinverse()

            assert h_i.shape[-1] == d_p.flatten().shape[0], "Tensor dimension mismatch"

            d2_p = (h_i @ d_p.flatten()).reshape(d_p.shape)

            dir_list.append(d2_p)

        return dir_list

    @torch.no_grad()
    def step(self, x, y, loss_fn, closure=None):
        """ """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_dict = dict(self._model.named_parameters())
        keys, values = zip(*param_dict.items())
        
        def eval_model(*input_params):
            # print(type(input_params))
            # print(type(input_params[0]))
            # print(input_params)
            out = functional_call(self._model, dict(zip(keys, input_params)), x)
            return loss_fn(out, y)

        self._h_list = []
        if not self.hessian_approx:
            self._h_list = torch.autograd.functional.hessian(eval_model, values, create_graph=True)
            self._h_list = [self._h_list[i][i] for i, _ in enumerate(self._h_list)]

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            lr = group["lr"]

            if self.hessian_approx:
                self._j_list = torch.autograd.functional.jacobian(eval_model, values, create_graph=False)
                for i, j in enumerate(self._j_list):
                    j = j.flatten()
                    h = torch.outer(j, j)
                    self._h_list.append(h)

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            h_list = [self._reshape_hessian(h) for h in self._h_list]

            self._apply_gradients(params=params_with_grad, d_p_list=d_p_list, h_list=h_list, lr=lr, eval_model=eval_model)

        return loss
