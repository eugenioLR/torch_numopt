import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd.functional import hessian
from torch.nn.utils.stateless import functional_call
from .second_order_optimizer import SecondOrderOptimizer
import copy

class LM_cpd(SecondOrderOptimizer):
    """
    Heavily inspired by https://github.com/hahnec/torchimize/blob/master/torchimize/optimizer/gna_opt.py
    """

    def __init__(self, params, lr, model, ld=1, hessian_approx = False):
        assert lr > 0, "Learning rate must be a positive number"

        super().__init__(params, {"lr": lr})

        self.hessian_approx = hessian_approx

        self._model = model
        self._params = self.param_groups[0]['params']
        self.ld = ld 
        self._j_list = []
        self._h_list = []
        self.prev_loss = None
    
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
            h_adjusted = h + self.ld * torch.eye(h.shape[0])
            # h_adjusted = h + self.ld * h.diagonal()
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
        
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

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
                self._h_list = [h.flatten(end_dim=len(self._h_list[i].shape)-len(d_p_list[i].shape)-1).flatten(start_dim=1) for i, h in enumerate(self._h_list)]
        
            self._apply_gradients(
                params_with_grad, 
                d_p_list,
                self._h_list,
                lr
            )
        
        # print(loss)
        # if self.prev_loss is not None and self.prev_loss:
        #     self.ld *= 10
        # else:
        #     self.ld /= 10
        # self.prev_loss = self._model(x).square().sum()

        
        return loss
    
    def update(self, model, loss):
        loss_val = loss.detach().item()

        if prev_loss is None or loss_val > self.prev_loss:
            pass
    
