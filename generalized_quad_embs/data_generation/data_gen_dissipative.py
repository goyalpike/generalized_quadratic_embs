import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from scipy.integrate import solve_ivp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#########################################################
################## DISSIPATIVE SYSTEMS ##################
#########################################################

class PendulumExampleDissipative:
    """It is pendulum example. It contains vector field and its Jacobian."""

    @staticmethod
    def vector_field(t, z):
        """Vector field of the pendulum example.

        Args:
            t (float): time
            z (float): state vector

        Returns:
            _type_: _description_
        """
        len_z = len(z)
        q, p = z[: len_z // 2], z[len_z // 2 :]
        dqdt = p
        dpdt = -np.sin(q) - 0.025*p
        dzdt = np.concatenate((dqdt, dpdt), axis=0)
        return dzdt

    @staticmethod
    def potential(q, p):
        """Defining potential of pendulum."""
        return (1 / 2) * p**2 + 1 - np.cos(q)


# Dissipative Lokta-Volterra
class LVExampleDissipative:
    """It is Lotka Volterra. It contains vector field and its Jacobian."""

    @staticmethod
    def vector_field(t, z):
        """Vector field of the nonlinear oscillator example.

        Args:
            t (float): time
            z (float): state vector

        Returns:
            _type_: _description_
        """
        len_z = len(z)
        q, p = z[: len_z // 2], z[len_z // 2 :]
        dqdt = 1 - np.exp(p) - 0.05*q
        dpdt = np.exp(q) - 2 - 0.05*p
        dzdt = np.concatenate((dqdt, dpdt), axis=0)
        return dzdt

    @staticmethod
    def potential(q, p):
        """Defining potential of LV."""
        return p - np.exp(p) + 2 * q - np.exp(q)
    

def get_data(params, dyn_model):
    """Generate data for a given dynamical model.

    Args:
        params (dict): It contans necessary parameters such as number of points, max_potential, etc.
        dyn_model (class): Dynamical model that has vector field and its Jacobian as methods.

    Returns:
        tuple: data and time
    """

    time = np.linspace(0, params.t_train, params.sample_size)

    for i in range(params.Nt):
        y0 = np.random.uniform(params.smpling_intl[0], params.smpling_intl[1], 2)

        while dyn_model.potential(y0[0], y0[1]) > params.max_potential:
            y0 = np.random.uniform(params.smpling_intl[0], params.smpling_intl[1], 2)

        # sol_train = Mid_point(dyn_model.vector_field, dyn_model.jacobian, y0, time)
        sol_train = solve_ivp(dyn_model.vector_field, t_span = [time[0], time[-1]], y0 = y0, t_eval = time)
        sol_train = sol_train.y

        if i == 0:
            x_data = sol_train.T
        else:
            x_data = np.concatenate((x_data, sol_train.T))

    return x_data, time


def data_loader(x_train, time, ode_model, params):
    """Given data, it creates datalaoders for training

    Args:
        x_train (float):  state
        time (float): time
        ode_model (function): vector field of a dynamical model
        params (dataclass): containing necessary parameters

    Returns:
        _type_: _description_
    """
    # Convert the data
    dxdt = ode_model(time, x_train.T).T
    dxdt = torch.from_numpy(dxdt)
    x_train = torch.tensor(x_train, dtype=torch.float64, requires_grad=True)

    train_ds = TensorDataset(x_train.to(device), dxdt.to(device))
    train_dl = DataLoader(train_ds, batch_size=params.batch_size, shuffle=True)
    return train_dl


