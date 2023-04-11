"""
Contains all parts of pyTorch based machine learning model.
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ...utils import maybe_progress
from .twice_differentiable import TwiceDifferentiable

__all__ = [
    "TorchTwiceDifferentiable",
]
logger = logging.getLogger(__name__)


def flatten_gradient(grad) -> torch.Tensor:
    """
    Simple function to flatten a pyTorch gradient for use in subsequent calculation
    """
    return torch.cat([el.reshape(-1) for el in grad])


def solve_linear(
    model: TwiceDifferentiable,
    training_data: DataLoader,
    b: torch.Tensor,
    lam: float = 0,
    progress: bool = True,
) -> torch.Tensor:
    """Given a model and training data, it uses conjugate gradient to calculate the
    inverse of the HVP. More precisely, it finds x s.t. $Hx = b$, with $H$ being
    the model hessian.

    :param model: A model wrapped in the TwiceDifferentiable interface.
    :param training_data: A DataLoader containing the training data.
    :param b: a vector or matrix
    :param lam: regularization of the hessian
    :param progress: If True, display progress bars.

    :return: An array that solves the inverse problem,
        i.e. it returns $x$ such that $Ax = b$
    """
    all_x, all_y = [], []
    for x, y in training_data:
        all_x.append(x)
        all_y.append(y)
    all_x = cat(all_x)
    all_y = cat(all_y)
    matrix = model.hessian(all_x, all_y, progress) + lam * identity_tensor(
        model.num_params()
    )
    return torch.linalg.solve(matrix, b.T).T


def solve_batch_cg(
    model: TwiceDifferentiable,
    training_data: DataLoader,
    b: torch.Tensor,
    lam: float = 0,
    inversion_method_kwargs: Dict[str, Any] = {},
    progress: bool = True,
) -> torch.Tensor:
    """
    Given a model and training data, it uses conjugate gradient to calculate the
    inverse of the HVP. More precisely, it finds x s.t. $Hx = b$, with $H$ being
    the model hessian. For more info:
    https://en.wikipedia.org/wiki/Conjugate_gradient_method

    :param model: A model wrapped in the TwiceDifferentiable interface.
    :param training_data: A DataLoader containing the training data.
    :param b: a vector or matrix
    :param lam: regularization of the hessian
    :param inversion_method_kwargs: kwargs to pass to the inversion method
    :param progress: If True, display progress bars.

    :return: A matrix of shape [NxP] with each line being a solution of $Ax=b$.
    """
    total_grad_xy = 0
    total_points = 0
    for x, y in maybe_progress(training_data, progress, desc="Batch Train Gradients"):
        grad_xy, _ = model.grad(x, y)
        total_grad_xy += grad_xy * len(x)
        total_points += len(x)
    backprop_on = model.parameters()
    reg_hvp = lambda v: mvp(total_grad_xy / total_points, v, backprop_on) + lam * v
    batch_cg = torch.zeros_like(b)
    for idx, bi in enumerate(maybe_progress(b, progress, desc="Conjugate gradient")):
        bi_cg, _ = solve_cg(reg_hvp, bi, **inversion_method_kwargs)
        batch_cg[idx] = bi_cg
    return batch_cg


def solve_cg(
    hvp: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    rtol: float = 1e-7,
    atol: float = 1e-7,
    maxiter: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Conjugate gradient solver for the Hessian vector product

    :param hvp: a Callable Hvp, operating with tensors of size N
    :param b: a tensor of shape [N]
    :param x0: initial guess for hvp
    :param rtol: maximum relative tolerance of result
    :param atol: absolute tolerance of result
    :param maxiter: maximum number of iterations. If None, defaults to 10*len(y)
    """
    if x0 is None:
        x0 = torch.clone(b)
    if maxiter is None:
        maxiter = len(b) * 10

    y_norm = torch.sum(torch.matmul(b, b)).item()
    stopping_val = max([rtol**2 * y_norm, atol**2])

    x = x0
    p = r = (b - hvp(x)).squeeze()
    gamma = torch.sum(torch.matmul(r, r)).item()
    optimal = False

    for k in range(maxiter):
        Ap = hvp(p).squeeze()
        alpha = gamma / torch.sum(torch.matmul(p, Ap)).item()
        x += alpha * p
        r -= alpha * Ap
        gamma_ = torch.sum(torch.matmul(r, r)).item()
        beta = gamma_ / gamma
        gamma = gamma_
        p = r + beta * p

        if gamma < stopping_val:
            optimal = True
            break

    info = {"niter": k, "optimal": optimal}
    return x, info


def solve_lissa(
    model: TwiceDifferentiable,
    training_data: DataLoader,
    b: torch.Tensor,
    lam: float = 0,
    progress: bool = True,
    maxiter: int = 1000,
    damp: float = 0,
    scale: float = 10,
    h0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    It uses LISSA, Linear time Stochastic Second-Order Algorithm, to calculate the
    inverse of the HVP. More precisely, it finds x s.t. $Hx = b$, with $H$ being
    the model hessian.
    This is done by iteratively approximating H through
    $$
    H^{-1}_{j+1} b = b + (I - H) \ H^{-1}_j b
    $$
    where I is the identity matrix. Additional damping and scaling factors are
    applied to help convergence. More info can be found in
    :footcite:t:`koh_understanding_2017`

    :param model: A model wrapped in the TwiceDifferentiable interface.
    :param training_data: A DataLoader containing the training data.
    :param b: a vector or matrix
    :param lam: regularization of the hessian
    :param progress: If True, display progress bars.
    :param maxiter: maximum number of iterations,
    :param damp: damping factor, defaults to 0 for no damping
    :param scale: scaling factor, defaults to 10
    :param h0: initial guess for hvp

    :return: A matrix of shape [NxP] with each line being a solution of $Ax=b$.
    """
    if h0 is None:
        h_estimate = torch.clone(b)
    else:
        h_estimate = h0
    shuffled_training_data = DataLoader(
        training_data.dataset, training_data.batch_size, shuffle=True
    )
    for _ in maybe_progress(range(maxiter), progress, desc="Lissa"):
        x, y = next(iter(shuffled_training_data))
        grad_xy, _ = model.grad(x, y)
        reg_hvp = lambda v: mvp(grad_xy, v, model.parameters()) + lam * v
        h_estimate = b + (1 - damp) * h_estimate - reg_hvp(h_estimate) / scale
        if torch.isnan(h_estimate).any():
            raise RuntimeError("NaNs in h_estimate. Increase scale or damp.")
    return h_estimate / scale


def as_tensor(a: Any, warn=True, **kwargs) -> torch.Tensor:
    """Converts an array into a torch tensor

    :param a: array to convert to tensor
    :param warn: if True, warns that a will be converted
    """
    if warn and not isinstance(a, torch.Tensor):
        logger.warning("Converting tensor to type torch.Tensor.")
    return torch.as_tensor(a, **kwargs)


def stack(a: Sequence[torch.Tensor], **kwargs) -> torch.Tensor:
    """Stacks a sequence of tensors into a single torch tensor"""
    return torch.stack(a, **kwargs)


def cat(a: Sequence[torch.Tensor], **kwargs) -> torch.Tensor:
    """Concatenates a sequence of tensors into a single torch tensor"""
    return torch.cat(a, **kwargs)


def einsum(equation, *operands) -> torch.Tensor:
    """Sums the product of the elements of the input :attr:`operands` along dimensions specified using a notation
    based on the Einstein summation convention.
    """
    return torch.einsum(equation, *operands)


def identity_tensor(dim: int) -> torch.Tensor:
    return torch.eye(dim, dim)


def mvp(
    grad_xy: torch.Tensor,
    v: torch.Tensor,
    backprop_on: torch.Tensor,
    progress: bool = False,
) -> torch.Tensor:
    """
    Calculates second order derivative of the model along directions v.
    This second order derivative can be selected through the backprop_on argument.

    :param grad_xy: an array [P] holding the gradients of the model
        parameters wrt input $x$ and labels $y$, where P is the number of
        parameters of the model. It is typically obtained through
        self.grad.
    :param v: An array ([DxP] or even one dimensional [D]) which
        multiplies the matrix, where D is the number of directions.
    :param progress: True, iff progress shall be printed.
    :param backprop_on: tensor used in the second backpropagation (the first
        one is along $x$ and $y$ as defined via grad_xy).
    :returns: A matrix representing the implicit matrix vector product
        of the model along the given directions. Output shape is [DxP] if
        backprop_on is None, otherwise [DxM], with M the number of elements
        of backprop_on.
    """
    device = grad_xy.device
    v = as_tensor(v, warn=False).to(device)
    if v.ndim == 1:
        v = v.unsqueeze(0)

    z = (grad_xy * Variable(v)).sum(dim=1)

    mvp = []
    for i in maybe_progress(range(len(z)), progress, desc="MVP"):
        mvp.append(
            flatten_gradient(autograd.grad(z[i], backprop_on, retain_graph=True))
        )
    mvp = torch.stack([grad.contiguous().view(-1) for grad in mvp])
    return mvp.detach()  # type: ignore


class TorchTwiceDifferentiable(TwiceDifferentiable[torch.Tensor, nn.Module]):
    def __init__(
        self,
        model: nn.Module,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        r"""
        :param model: A (differentiable) function.
        :param loss: :param loss: A differentiable scalar loss $L(\hat{y}, y)$,
               mapping a prediction and a target to a real value.
        :param device: device to use for computations. Defaults to cuda if available.
        """
        if model.training:
            logger.warning(
                "Passed model not in evaluation mode. This can create several issues in influence "
                "computation, e.g. due to batch normalization. Please call model.eval() before "
                "computing influences."
            )
        self.model = model.to(device)
        self.loss = loss
        self.device = device

    def parameters(self) -> List[torch.Tensor]:
        """Returns all the model parameters that require differentiating"""
        return [
            param for param in self.model.parameters() if param.requires_grad == True
        ]

    def num_params(self) -> int:
        """
        Get number of parameters of model f.
        :returns: Number of parameters as integer.
        """
        return sum([np.prod(p.size()) for p in self.parameters()])

    def split_grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        progress: bool = False,
    ) -> torch.Tensor:
        """
        Calculates gradient of model parameters wrt each $x[i]$ and $y[i]$ and then
        returns a array of size [N, P] with N number of points (length of x and y) and P
        number of parameters of the model.
        :param x: An array [NxD] representing the features $x_i$.
        :param y: An array [NxK] representing the predicted target values $y_i$.
        :param progress: True, iff progress shall be printed.
        :returns: An array [NxP] representing the gradients with respect to
        all parameters of the model.
        """
        x = as_tensor(x, warn=False).to(self.device).unsqueeze(1)
        y = as_tensor(y, warn=False).to(self.device)

        grads = []
        for i in maybe_progress(range(len(x)), progress, desc="Split Gradient"):
            grads.append(
                flatten_gradient(
                    autograd.grad(
                        self.loss(
                            torch.squeeze(self.model(x[i])),
                            torch.squeeze(y[i]),
                        ),
                        self.parameters(),
                    )
                ).detach()
            )

        return torch.stack(grads, axis=0)

    def grad(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates gradient of model parameters wrt the model parameters.
        :param x: A matrix [NxD] representing the features $x_i$.
        :param y: A matrix [NxK] representing the target values $y_i$.
        :param x_requires_grad: If True, the input $x$ is marked as requiring
            gradients. This is important for further differentiation on input
            parameters.
        :returns: A tuple where: \
            - first element is an array [P] with the gradients of the model. \
            - second element is the input to the model as a grad parameters. \
                This can be used for further differentiation. 
        """
        x = as_tensor(x, warn=False).to(self.device).requires_grad_(x_requires_grad)
        y = as_tensor(y, warn=False).to(self.device)

        loss_value = self.loss(torch.squeeze(self.model(x)), torch.squeeze(y))
        grad_f = torch.autograd.grad(loss_value, self.parameters(), create_graph=True)
        return flatten_gradient(grad_f), x

    def hessian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        progress: bool = False,
    ) -> torch.Tensor:
        """Calculates the explicit hessian of model parameters given data ($x$ and $y$).
        :param x: A matrix [NxD] representing the features $x_i$.
        :param y: A matrix [NxK] representing the target values $y_i$.
        :returns: the hessian of the model, i.e. the second derivative wrt. the model parameters.
        """
        x = x.to(self.device)
        y = y.to(self.device)
        grad_xy, _ = self.grad(x, y)
        return mvp(
            grad_xy,
            torch.eye(self.num_params(), self.num_params()),
            self.parameters(),
            progress,
        )
