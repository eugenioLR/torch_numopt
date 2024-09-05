import torch


class LineSearchMixin:
    def accept_step(self, params, new_params, step_dir, lr, loss, new_loss, grad, c1, c2, line_search_cond):
        """ """

        accepted = True

        dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(grad, step_dir)])

        if line_search_cond == "armijo":
            accepted = new_loss <= loss + c1 * lr * dir_deriv
        elif line_search_cond == "wolfe":
            new_grad = torch.autograd.grad(new_loss, new_params)
            new_dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir)])
            armijo = new_loss <= loss + c1 * lr * dir_deriv
            curv_cond = new_dir_deriv >= c2 * dir_deriv
            accepted = armijo and curv_cond
        elif line_search_cond == "strong-wolfe":
            new_grad = torch.autograd.grad(new_loss, new_params)
            new_dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir)])
            armijo = new_loss <= loss + c1 * lr * dir_deriv
            curv_cond = abs(new_dir_deriv) <= c2 * abs(dir_deriv)
            accepted = armijo and curv_cond
        elif line_search_cond == "goldstein":
            accepted = loss + (1 - c1) * lr * dir_deriv <= new_loss <= loss + c1 * lr * dir_deriv
        else:
            raise ValueError(f"Line search condition {line_search_cond} does not exist.")

        return accepted

    @torch.enable_grad()
    def backtrack_wolfe(self, params, step_dir, grad, lr_init, eval_model, c1, c2, tau, line_search_cond="armijo"):
        """ """

        lr = lr_init

        loss = eval_model(*params)

        new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
        new_loss = eval_model(*new_params)

        while not self.accept_step(params, new_params, step_dir, lr, loss, new_loss, grad, c1, c2, line_search_cond):
            lr *= tau

            # Evaluate model with new lr
            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)

            if lr <= 1e-10:
                break

        return new_params
