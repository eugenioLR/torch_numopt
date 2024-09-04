import torch


def fix_stability(mat):
    """
    Procedure to adjust a matrix by adding to the diagonal an very small value to avoid numerical
    instability problems.
    """

    return mat + torch.eye(mat.shape[0], device=mat.device) * torch.finfo(mat.dtype).eps


def pinv_svd_trunc(mat, thresh=1e-4):
    """
    Procedure to calculate the pseudoinverse of a matrix by using truncated SVD in order to mantain
    numerical stability.
    """

    U, S, Vt = torch.linalg.svd(mat)

    S_tresh = S < thresh

    S_inv_trunc = 1.0 / S
    S_inv_trunc[S_tresh] = 0

    return Vt.T @ torch.diag(S_inv_trunc) @ U.T
