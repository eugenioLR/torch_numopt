import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd.functional import hessian
from torch.func import functional_call
from .second_order_optimizer import SecondOrderOptimizer, fix_stability, pinv_svd_trunc
import warnings
from copy import copy, deepcopy

class LM(SecondOrderOptimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
    """

    def __init__(self, params, lr, model, ld=1, use_diagonal = True, hessian_approx = False, debug_stability = True):
        assert lr > 0, "Learning rate must be a positive number" 

        super().__init__(params, {"lr": lr})

        self.hessian_approx = hessian_approx

        self._model = model
        self._params = self.param_groups[0]['params']
        self.ld = ld 
        self._j_list = []
        self._h_list = []
        self.prev_loss = None
        self._prev_params = deepcopy(self.param_groups[0]['params'])
        self.use_diagonal = use_diagonal
        self.debug_stability = debug_stability
    
    def _apply_gradients(self, params, d_p_list, h_list, lr):
        """
        """

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
                h_adjusted = h + self.ld * adjustment
                
                # Use truncated SVD pseudoinverse to address numerical instability
                h_i = pinv_svd_trunc(h_adjusted)
            else:
                adjustment = torch.eye(h.shape[0], device=h.device)
                h_adjusted = h + self.ld * adjustment
                h_i = h_adjusted.pinverse()

            assert h_i.shape[-1] == d_p.flatten().shape[0], "Tensor dimension mismatch"

            d2_p = h_i.matmul(d_p.flatten()).reshape(d_p_list[i].shape)
            param.add_(d2_p, alpha=-lr)

    @torch.no_grad()
    def step(self, x, closure=None):
        """
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        parameters = dict(self._model.named_parameters())
        keys, values = zip(*parameters.items())

        self._h_list = []

        if self.hessian_approx:
            raise Exception("Not implemented yet, bruh.")
        else:
            def func(*params):
                out = functional_call(self._model, {n: p for n, p in zip(keys, params)}, x)
                return out.square().sum()
            self._h_list = torch.autograd.functional.hessian(func, tuple(self._model.parameters()), create_graph=True)
            self._h_list = [self._h_list[i][i] for i, _ in enumerate(self._h_list)]
            
        
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

            h_list = [h.flatten(end_dim=len(self._h_list[i].shape)-len(d_p_list[i].shape)-1).flatten(start_dim=1) for i, h in enumerate(self._h_list)]
        
            self._apply_gradients(
                params_with_grad, 
                d_p_list,
                h_list,
                lr
            )
        
        return loss
    
    def update(self, loss):
        loss_val = loss.detach().item()

        if self.prev_loss is None or loss_val > self.prev_loss:
            self._params = self._prev_params
            # self.ld *= 10
        else:
            self.prev_loss = loss_val
            self._prev_params = deepcopy(self._params)
            # self.ld /= 10
        
