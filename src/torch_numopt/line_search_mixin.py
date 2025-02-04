import torch


class LineSearchMixin:
    """
    Mixin to add a line search procedure to an optmization algorithm.
    """

    @torch.enable_grad()
    def accept_step(self, params, new_params, step_dir, lr, loss, new_loss, grad, c1, c2, line_search_cond):
        """ """

        accepted = True

        dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(grad, step_dir)])

        match line_search_cond:
            case "armijo":
                accepted = new_loss <= loss + c1 * lr * dir_deriv
            case "wolfe":
                new_grad = torch.autograd.grad(new_loss, new_params)
                new_dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir)])
                armijo = new_loss <= loss + c1 * lr * dir_deriv
                curv_cond = new_dir_deriv >= c2 * dir_deriv
                accepted = armijo and curv_cond
            case "strong-wolfe":
                new_grad = torch.autograd.grad(new_loss, new_params)
                new_dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(new_grad, step_dir)])
                armijo = new_loss <= loss + c1 * lr * dir_deriv
                curv_cond = abs(new_dir_deriv) <= c2 * abs(dir_deriv)
                accepted = armijo and curv_cond
            case "goldstein":
                accepted = loss + (1 - c1) * lr * dir_deriv <= new_loss <= loss + c1 * lr * dir_deriv
            case _:
                raise ValueError(f"Line search condition {line_search_cond} does not exist.")

        return accepted

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

    
    def interpolate_quadratic(self, params, step_dir, grad, lr_init, eval_model, c1, c2, line_search_cond="armijo"):
        dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(grad, step_dir)])

        loss = eval_model(*params)
        lr = lr_init

        new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
        new_loss = eval_model(*new_params)

        while not self.accept_step(params, new_params, step_dir, lr, loss, new_loss, grad, c1, c2, line_search_cond):
            if lr == 0:
                break

            lr = - 0.5 * (dir_deriv * lr ** 2) / (new_loss - loss - dir_deriv * lr)

            new_params = tuple(p - lr * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)

        return new_params
    
    def interpolate_cubic(self, params, step_dir, grad, lr_init, eval_model, c1, c2, line_search_cond="armijo"):
        dir_deriv = sum([torch.dot(p_grad.flatten(), p_step.flatten()) for p_grad, p_step in zip(grad, step_dir)])

        loss = eval_model(*params)
        lr_0 = lr_init

        # Calculate first interpolation point
        prev_params = tuple(p - lr_0 * p_step for p, p_step in zip(params, step_dir))
        prev_loss = eval_model(*prev_params)

        if self.accept_step(params, prev_params, step_dir, lr_0, loss, prev_loss, grad, c1, c2, line_search_cond):
            return prev_params
        
        # print()
        # Calculate second interpolation point
        lr_1 = - 0.5 * (dir_deriv * lr_0 ** 2) / (prev_loss - loss - dir_deriv * lr_0)
        # print(lr_1)

        new_params = tuple(p - lr_1 * p_step for p, p_step in zip(params, step_dir))
        new_loss = eval_model(*new_params)

        # print()
        while not self.accept_step(params, new_params, step_dir, lr_1, loss, new_loss, grad, c1, c2, line_search_cond):
            if lr_0 == 0 or lr_1 == 0 or lr_1 == lr_0:
                break
            factor =  1 / ((lr_0 * lr_1)**2 * (lr_1 - lr_0))
            aux_mat = torch.Tensor([[lr_0**2, -lr_1**2], [-lr_0**3, lr_1**3]])
            aux_vec = torch.Tensor([new_loss - loss - dir_deriv*lr_1, prev_loss - loss - dir_deriv*lr_0])
            a, b = factor.cpu()*torch.matmul(aux_mat, aux_vec)
            
            lr_0 = lr_1
            lr_1 = (-b + torch.sqrt(torch.abs(b**2 - 3*a*dir_deriv)))/(3*a)
            # print(float(lr_0), float(lr_1), a, b, b**2 - 3*a*dir_deriv)

            prev_loss = new_loss
            new_params = tuple(p - lr_1 * p_step for p, p_step in zip(params, step_dir))
            new_loss = eval_model(*new_params)

        return new_params

    
    
