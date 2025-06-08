import os, random
import torch
import numpy as np
import torch.autograd.forward_ad as fwAD
import matplotlib.pyplot as plt
# from integrator import Mid_point
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def kron(x, y):
    t = torch.einsum("ab,ad->abd", [x, y]).view(x.size(0), x.size(1) * y.size(1))
    return t


def jacobian_bcw(y, x):
    diagonal_matrix = torch.eye(y.shape[1]).double().to(device)
    cotangents = torch.stack(
        [cotangent.repeat(x.shape[0], 1) for cotangent in diagonal_matrix]
    )
    rev_jacobian = []
    for cotangent in cotangents:
        (jacobian_row,) = torch.autograd.grad(
            outputs=(y,), inputs=(x,), grad_outputs=(cotangent,), create_graph=True
        )
        rev_jacobian.append(jacobian_row)
    # jacobian: [M, B, N]
    jacobian = torch.stack(rev_jacobian)
    # jacobian: [B, M, N]
    jacobian = jacobian.transpose(1, 0)
    return jacobian


def jacobian_frwd(fun, x):
    # https://github.com/leimao/PyTorch-Automatic-Differentiation/blob/main/autograd_weights.py
    diagonal_matrix = torch.eye(x.shape[1]).double().to(device)
    tangents = torch.stack(
        [cotangent.repeat(x.shape[0], 1) for cotangent in diagonal_matrix]
    )

    # Method 1
    # Use PyTorch autograd forward mode + `for` loop.
    fwd_jacobian = []
    with fwAD.dual_level():
        # N forward pass
        for tangent in tangents:
            dual_input = fwAD.make_dual(x, tangent)
            # Tensors that do not not have an associated tangent are automatically
            # considered to have a zero-filled tangent of the same shape.
            dual_output = fun(dual_input)
            # Unpacking the dual returns a namedtuple with ``primal`` and ``tangent``
            # as attributes
            jacobian_column = fwAD.unpack_dual(dual_output).tangent
            fwd_jacobian.append(jacobian_column)
    fwd_jacobian = torch.stack(fwd_jacobian).permute(1, 2, 0)
    return fwd_jacobian


def tensor_J(batch_size, n):
    J = None
    J = torch.eye(n, dtype=torch.float64)
    J = torch.cat([J[n // 2 :], -J[: n // 2]])
    J = torch.unsqueeze(J, dim=0)
    J = J.repeat(batch_size, 1, 1)
    return J


def can_tp(x, y):
    t = torch.einsum("abc,ace->abe", [x, y])
    return t


def get_data(params, model, jac_model, potential):
    t_span = [0, params.t_train]
    t_train = np.linspace(t_span[0], t_span[1], params.Nt)
    z_train = []

    for i in range(params.sample_size):
        y0 = np.random.uniform(params.smpling_intl[0], params.smpling_intl[1], 2)

        while potential(y0[0], y0[1]) > params.max_potential:
            y0 = np.random.uniform(params.smpling_intl[0], params.smpling_intl[1], 2)

        #         print('initial condition',y0)

        sol_train = Mid_point(model, jac_model, y0, t_train)
        if i == 0:
            z_train = sol_train.T
        else:
            z_train = np.concatenate((z_train, sol_train.T))
    return y0, z_train, t_train, sol_train


def data_loader(z_train, t_train, ode_model, params):
    # Convert the data
    dzdt = ode_model(t_train, z_train.T).T
    dzdt = torch.from_numpy(dzdt)
    z_train = torch.tensor(z_train, dtype=torch.float64, requires_grad=True)

    train_ds = TensorDataset(z_train.to(device), dzdt.to(device))
    train_dl = DataLoader(train_ds, batch_size=params.batch_size, shuffle=True)
    return train_dl


def reproducibility_seed(seed: int) -> None:
    """To set seed for random variables."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def define_color_method(params):
    color_method_pair = {
        "linear": (3, "s-linear-embs"),
        "cubic": (1, "s-cubic-embs"),
        "quad": (2, "quad-embs"),
        "linear_nostability": (4, "linear-embs"),
    }

    if params.confi_model in color_method_pair:
        return color_method_pair[params.confi_model]
    else:
        return (5, "quad-OpInf")
        color_idx = 5


def compute_derivative(dx, y):

    derivative = []
    n = len(y)
    # Calculate the derivative using the 5-order stencil
    for i in range(n):
        if i < 2:
            derivative.append(
                (
                    -25 * y[i]
                    + 48 * y[i + 1]
                    - 36 * y[i + 2]
                    + 16 * y[i + 3]
                    - 3 * y[i + 4]
                )
                / (12 * dx)
            )
        elif i >= n - 2:
            derivative.append(
                (
                    25 * y[i]
                    - 48 * y[i - 1]
                    + 36 * y[i - 2]
                    - 16 * y[i - 3]
                    + 3 * y[i - 4]
                )
                / (12 * dx)
            )
        else:
            derivative.append(
                (-y[i + 2] + 8 * y[i + 1] - 8 * y[i - 1] + y[i - 2]) / (12 * dx)
            )

    return derivative
    
    
# The following code is taken from
#  https://github.com/Willcox-Research-Group/rom-operator-inference-Python3

def ddt_uniform(states, dt, order=2):
    """Approximate the time derivatives for a chunk of snapshots that are
    uniformly spaced in time.
    Parameters
    ----------
    states : (n, k) ndarray
        States to estimate the derivative of. The jth column is a snapshot
        that corresponds to the jth time step, i.e., states[:, j] = x(t[j]).
    dt : float
        The time step between the snapshots, i.e., t[j+1] - t[j] = dt.
    order : int {2, 4, 6}
        The order of the derivative approximation.
        See https://en.wikipedia.org/wiki/Finite_difference_coefficient.
    Returns
    -------
    ddts : (n, k) ndarray
        Approximate time derivative of the snapshot data. The jth column is
        the derivative dx / dt corresponding to the jth snapshot, states[:, j].
    """
    # Check dimensions and input types.
    if states.ndim != 2:
        raise ValueError("states must be two-dimensional")
    if not np.isscalar(dt):
        raise TypeError("time step dt must be a scalar (e.g., float)")

    if order == 2:
        return np.gradient(states, dt, edge_order=2, axis=1)

    Q = states
    ddts = np.empty_like(states)
    n, k = states.shape
    if order == 4:
        # Central difference on interior.
        ddts[:, 2:-2] = (Q[:, :-4] - 8 * Q[:, 1:-3] + 8 * Q[:, 3:-1] - Q[:, 4:]) / (
            12 * dt
        )

        # Forward / backward differences on the front / end.
        for j in range(2):
            ddts[:, j] = _fwd4(Q[:, j : j + 5].T, dt)  # Forward
            ddts[:, -j - 1] = -_fwd4(Q[:, -j - 5 : k - j].T[::-1], dt)  # Backward

    elif order == 6:
        # Central difference on interior.
        ddts[:, 3:-3] = (
            -Q[:, :-6]
            + 9 * Q[:, 1:-5]
            - 45 * Q[:, 2:-4]
            + 45 * Q[:, 4:-2]
            - 9 * Q[:, 5:-1]
            + Q[:, 6:]
        ) / (60 * dt)

        # Forward / backward differences on the front / end.
        for j in range(3):
            ddts[:, j] = _fwd6(Q[:, j : j + 7].T, dt)  # Forward
            ddts[:, -j - 1] = -_fwd6(Q[:, -j - 7 : k - j].T[::-1], dt)  # Backward

    else:
        raise NotImplementedError(
            f"invalid order '{order}'; " "valid options: {2, 4, 6}"
        )

    return ddts


# Finite difference stencils ==================================================
def _fwd4(y, dt):
    """Compute the first derivative of a uniformly-spaced-in-time array with a
    fourth-order forward difference scheme.
    Parameters
    ----------
    y : (5, ...) ndarray
        Data to differentiate. The derivative is taken along the first axis.
    dt : float
        Time step (the uniform spacing).
    Returns
    -------
    dy0 : float or (...) ndarray
        Approximate derivative of y at the first entry, i.e., dy[0] / dt.
    """
    return (-25 * y[0] + 48 * y[1] - 36 * y[2] + 16 * y[3] - 3 * y[4]) / (12 * dt)


def _fwd6(y, dt):
    """Compute the first derivative of a uniformly-spaced-in-time array with a
    sixth-order forward difference scheme.
    Parameters
    ----------
    y : (7, ...) ndarray
        Data to differentiate. The derivative is taken along the first axis.
    dt : float
        Time step (the uniform spacing).
    Returns
    -------
    dy0 : float or (...) ndarray
        Approximate derivative of y at the first entry, i.e., dy[0] / dt.
    """
    return (
        -147 * y[0]
        + 360 * y[1]
        - 450 * y[2]
        + 400 * y[3]
        - 225 * y[4]
        + 72 * y[5]
        - 10 * y[6]
    ) / (60 * dt)
