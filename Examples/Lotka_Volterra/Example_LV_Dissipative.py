#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import savemat

from generalized_quad_embs.data_generation.data_gen import data_loader
import generalized_quad_embs.data_generation.data_gen as data_gen
import generalized_quad_embs.data_generation.data_gen_dissipative as data_gen_dissipative
import generalized_quad_embs.modules_quad_stable as module
import generalized_quad_embs.plots_helper as plot
import generalized_quad_embs.utils as utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("No GPU found!")
else:
    print("Great, a GPU is there")
print("=" * 50)


# Plotting setting
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

plt.rc("font", size=20)  # controls default text size
plt.rc("axes", titlesize=20)  # fontsize of the title
plt.rc("axes", labelsize=20)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=20)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=20)  # fontsize of the y tick labels
plt.rc("legend", fontsize=15)  # fontsize of the legend

# Define the parameters
@dataclass
class Parameters:
    """It contain necessary parameters for this example."""

    t_train: float = 10.0  # Training final time
    t_test: float = 10.0  # Testing final time
    Nt: int = 10  # Number initial conditions of training samples
    Ntest: int = 100  # Number initial conditions of testing samples
    canonical_dim: int = 2  # canonical dimension
    latent_dim: int = None  # latent canonical dimensional
    hidden_dim: int = 16  # number of neurons in a hidden layer
    smpling_intl = [-1.5, 1.5]  # sampling interval
    sample_size: int = 200  # number of samples in a given training time interval
    batch_size: int = 64  # batch size
    max_potential = 4  # max potential to generate training data
    learning_rate: float = 3e-3  # Learning rate
    encoder: str = "MLP"  # type of neural network
    confi_model: str = None  # model configuration
    epoch: int = None  # number of epochs which are externally controlled
    path: str = None  # path where the results will be save and it is also externally controlled
    loss_weights: tuple = (1.0, 1.0)


# Prepare learned models for integration
def learned_model(t, x):
    """It yields time-derivative of x at time t.
    It is obtained throught the time-derivative of Hamiltonian function.

    Args:
        t (float): time
        x (float): state variable containing position and momenta.

    Returns:
        float: time-derivative of x
    """
    x = torch.tensor(
        x.reshape(-1, params.latent_dim), dtype=torch.float64, requires_grad=True
    ).to(device)
    y = vf.vector_field(x)
    y = y.detach()
    return y.cpu().numpy()


def plotting_saving_plots_phasespace(
    gt_sol,
    decoded_latent_sol,
    color_idx,
    method_name,
    idx,
    path="",
    closing_plts=True,
):

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(
        gt_sol.T[:, 0],
        gt_sol.T[:, 1],
        linestyle="None",
        marker="o",
        markevery=10,
        color="k",
    )
    plt.xlabel("q")
    plt.ylabel("p")

    ax.plot(
        decoded_latent_sol.T[:, 0], decoded_latent_sol.T[:, 1], color=colors[color_idx]
    )
    plt.title(method_name)
    plt.tight_layout()

    fig.savefig(
        path + f"plot_learned_ps_{idx}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )

    if closing_plts:
        plt.close("all")



# For reproducibility
utils.reproducibility_seed(seed=42)

params = Parameters()

# Taking external inputs
parser = argparse.ArgumentParser()

parser.add_argument(
    "--confi_model",
    type=str,
    default="linear",
    choices={"linear", "quad", "linear_nostability", "quad_opinf"},
    help="Enforcing model hypothesis",
)

parser.add_argument("--epochs", type=int, default=1200, help="Number of epochs")
parser.add_argument("--latent_dim", type=int, default=2, help="latent_dim")


args = parser.parse_args()

params.confi_model = args.confi_model
params.epoch = args.epochs
params.latent_dim = args.latent_dim
params.path = "./../Results/LV/" + params.confi_model + "/"

print('='*50)
print('==== Printing parameters setting')
print(params)
print('='*50)

if not os.path.exists(params.path):
    os.makedirs(params.path)
    print("The new directory is created as " + params.path)


# Generating data
dyn_model_dissipative = data_gen_dissipative.LVExampleDissipative()

# Obtaining the data
x_train, t_train = data_gen_dissipative.get_data(params, dyn_model_dissipative)
# Create data loaders. Note that the data contains both state and derivative information.
train_dl_dissipative = data_gen.data_loader(
    x_train, t_train, dyn_model_dissipative.vector_field, params
)


models = module.network_models(params)
optim = torch.optim.Adam(
    [
        {
            "params": models["ae"].parameters(),
            "lr": params.learning_rate,
            "weight_decay": 1e-5,
        },
        {
            "params": models["vf"].parameters(),
            "lr": params.learning_rate,
            "weight_decay": 1e-5,
        },
    ]
)



# Obtain Hamiltonian function, autoencoder, transformation and loss error
models, err_t = module.train_quad(models, train_dl_dissipative, optim, params)


# Extracting autoencoder and hnn (hamiltonian)
autoencoder, vf = models["ae"], models["vf"]



# Testing the models for different initial conditions

utils.reproducibility_seed(seed=100)
color_idx, method_name = utils.define_color_method(params)
learned_sol = []
ground_truth_sol = []
track_err = []


unstable_conf = 0
for i in range(params.Ntest):

    t_span = [0, 3 * params.t_test]
    t = np.linspace(t_span[0], t_span[1], 20 * params.sample_size)

    # define initial condition
    y0_test = np.random.uniform(params.smpling_intl[0], params.smpling_intl[1], 2)

    while (
        np.abs(dyn_model_dissipative.potential(y0_test[0], y0_test[1]))
        > params.max_potential
    ):
        y0_test = np.random.uniform(params.smpling_intl[0], params.smpling_intl[1], 2)

    y0_test = torch.tensor(y0_test, dtype=torch.float64)
    encoded_initial = autoencoder.encode(y0_test.to(device)).detach().cpu()

    print("Testing the models for initial conditon:", y0_test.numpy())
    print("Encoded initial condition:              ", encoded_initial.numpy())
    print("=" * 50)

    gt_sol = solve_ivp(
        dyn_model_dissipative.vector_field, t_span=[t[0], t[-1]], y0=y0_test, t_eval=t
    )
    gt_sol = gt_sol.y

    ground_truth_sol.append(gt_sol)

    latent_sol = solve_ivp(
        learned_model, t_span=[t[0], t[-1]], y0=encoded_initial.numpy(), t_eval=t
    )
    latent_sol = latent_sol.y

    if latent_sol.shape[-1] != len(t):
        unstable_conf += 1
        print("WARNING! Instability observed!")
        decoded_latent_sol = gt_sol*np.nan
        learned_sol.append(decoded_latent_sol)
        temp_err = gt_sol - decoded_latent_sol
        track_err.append(temp_err)
        continue

    latent_sol = torch.tensor(latent_sol, dtype=torch.float64).to(device)
    decoded_latent_sol = autoencoder.decode(latent_sol.T).detach().T.cpu().numpy()
    learned_sol.append(decoded_latent_sol)

    plot.plotting_saving_plots(
        t,
        gt_sol,
        decoded_latent_sol,
        color_idx=color_idx,
        method_name=method_name,
        idx=i,
        path=params.path,
        closing_plts=False,
    )

    plotting_saving_plots_phasespace(
        gt_sol,
        decoded_latent_sol,
        color_idx=color_idx,
        method_name=method_name,
        idx=i,
        path=params.path,
        closing_plts=False,
    )

    temp_err = gt_sol - decoded_latent_sol
    track_err.append(temp_err)

print("=" * 50)
print(f"=== Number of unstable configs: {unstable_conf}")
print("=" * 50)


learned_sol_stack = np.stack(learned_sol, axis=2)
ground_truth_sol_stack = np.stack(ground_truth_sol, axis=2)


savemat(
    params.path + "sol_trajectories.mat",
    {
        "learned_sol": learned_sol_stack,
        "ground_truth_sol": ground_truth_sol_stack,
        "t": t,
        "err": track_err,
        "unstable_conf": unstable_conf,
    },
)
