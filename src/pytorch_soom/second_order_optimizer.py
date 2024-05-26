import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd.functional import hessian
from torch.nn.utils.stateless import functional_call
import copy

class SecondOrderOptimizer(Optimizer):
    """
    Class for Optimization methods using second derivative information.
    """

    def step(self, x, closure=None):
        """
        """

    def update(self, loss):
        """
        """

def fix_stability(mat):
    diag_vec = mat.diagonal() + torch.finfo(mat.dtype).eps * 1
    mat.as_strided([mat.size(0)], [mat.size(0) + 1]).copy_(diag_vec)
    return mat

def pinv_svd_trunc(mat, thresh=1e-4):
    U, S, Vt = torch.linalg.svd(mat)

    S_tresh = S < thresh

    S_inv_trunc = 1.0/S
    S_inv_trunc[S_tresh] = 0
    
    return Vt.T @ torch.diag(S_inv_trunc) @ U.T


