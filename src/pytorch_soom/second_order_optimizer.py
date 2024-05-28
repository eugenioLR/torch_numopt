import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd.functional import hessian
from torch.nn.utils.stateless import functional_call
from functools import reduce
import copy


class SecondOrderOptimizer(Optimizer):
    """
    Class for Optimization methods using second derivative information.
    """

    def step(self, x, closure=None):
        """ """

    def update(self, loss):
        """ """

    @staticmethod
    def _reshape_hessian(hess):
        if len(hess.shape) == 2:
            return hess

        if len(hess.shape) % 2 != 0:
            raise ValueError("Hessian has a weird ass shape.")

        # Divide shape in two halves, multiply each half to get total size
        new_shape = (
            reduce(lambda x, y: x * y, hess.size()[hess.dim() // 2 :]),
            reduce(lambda x, y: x * y, hess.size()[: hess.dim() // 2]),
        )

        assert new_shape[0] == new_shape[1], "Something weird happened with the hessian size"

        return hess.reshape(new_shape)


def fix_stability(mat):
    # diag_vec = mat.diagonal() + torch.finfo(mat.dtype).eps * 1
    
    # mat.as_strided([mat.size(0)], [mat.size(0) + 1]).copy_(diag_vec)
    return mat + torch.eye(mat.shape[0]) * torch.finfo(mat.dtype).eps 


def pinv_svd_trunc(mat, thresh=1e-4):
    U, S, Vt = torch.linalg.svd(mat)

    S_tresh = S < thresh

    S_inv_trunc = 1.0 / S
    S_inv_trunc[S_tresh] = 0

    return Vt.T @ torch.diag(S_inv_trunc) @ U.T
