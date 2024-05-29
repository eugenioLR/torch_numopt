# Pytorch SOOM

Implementation of second order optimization methods for Neural Networks.

Due to computational constraints, these methods are to be used with small Neural Networks as they require $O(p^3)$ space for a network with $p$ parameters.

## References
[relevant paper](https://iopscience.iop.org/article/10.1088/1757-899X/495/1/012003/pdf)

## Planned optimizers

- [x] Newton-Raphson
- [x] Gauss-Newton
- [x] Levemberg-Marquard (LM)
- [x] Approximate Greatest Descent (AGD)
- [ ] Conjugate Gradient
- [ ] Quasi-Newton (LBFGS already in pytorch)
- [ ] Hessian-free / truncated Newton
