# Pytorch SOOM

Implementation of second order optimization methods for Neural Networks.

Due to computational constraints, these methods are to be used with small Neural Networks as they require $O(p^3)$ space for a network with $p$ parameters.

## References
[relevant paper](https://iopscience.iop.org/article/10.1088/1757-899X/495/1/012003/pdf)

## Planned optimizers

- [ ] Newton
- [ ] Conjugate Gradient
- [ ] Gauss-Newton
- [ ] Levemberg-Marquard (LM)
- [ ] Quasi-Newton (LBFGS already in pytorch)
- [ ] Approximate Greatest Descent (AGD)
- [ ] Hessian-free / truncated Newton