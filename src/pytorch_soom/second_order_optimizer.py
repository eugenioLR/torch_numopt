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