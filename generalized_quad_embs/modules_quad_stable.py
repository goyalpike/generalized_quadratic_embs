import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import generalized_quad_embs.utils as util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def jacobian(y, x):
    """It aims to compute the derivate of the output with respect to input.

    Args:
        y (float): Output
        x (float): Input

    Returns:
        dy/dx (float): Derivative of output w.r.t. input
    """
    batchsize = x.shape[0]
    dim = y.shape[1]
    res = torch.zeros(x.shape[0], y.shape[1], x.shape[1]).to(x)
    for i in range(dim):
        res[:, i, :] = torch.autograd.grad(
            y[:, i], x, grad_outputs=torch.ones(batchsize).to(x), create_graph=True
        )[0].reshape(res[:, i, :].shape)
    return res


def batch_mtxproduct(y, x):
    """It does batch wise matrix-matrix product. The first dimension is the batch.

    Args:
        y (float): A matrix of size a x b x c
        x (float): A matrix of size a x c x e

    Returns:
        y*x (float): Batch wise product of y and x
    """
    x = x.unsqueeze(dim=-1)
    return torch.einsum("abc,ace->abe", [y, x]).view(y.size(0), y.size(1))


def permutation_tensor(n):
    """Generating symplectic matrix

    Args:
        n (int): dimension of the system

    Returns:
        float (matrix): Symplectic matrix
    """
    J = None
    J = torch.eye(n)
    J = torch.cat([J[n // 2 :], -J[: n // 2]])

    return J


class AutoEncoderLinear(nn.Module):
    """It is meant to have linear encoder and decoder. Precisely, they both are identity.
    It is inhreited from nn.Module.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, x):
        """Encoder function

        Args:
            x (float): state

        Returns:
            float: encoded state
        """
        return x

    def decode(self, z):
        """Decoder function

        Args:
            z (float): encoded state

        Returns:
            float: decoded state
        """
        return z

    def forward(self, x):
        """Classical auto-encoder

        Args:
            x (float): state

        Returns:
            float: state via auto-encoder
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class AutoEncoderMLPTabular(nn.Module):
    """It is meant to have nonlinear encoder and decoder. Precisely, here we have hardcoded three hidden layers.
    It is inhreited from nn.Module.
    """

    def __init__(self, dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)

        self.linear5 = torch.nn.Linear(latent_dim, hidden_dim)
        self.linear6 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = torch.nn.Linear(hidden_dim, dim)

    def encode(self, x):
        """Encoder function

        Args:
            x (float): state

        Returns:
            float: encoded state
        """
        h = self.linear1(x)
        h = h + F.silu(self.linear2(h))
        h = h + F.silu(self.linear3(h))
        return self.linear4(h)

    def decode(self, z):
        """Decoder function

        Args:
            z (float): encoded state

        Returns:
            float: decoded state
        """
        h = self.linear5(z)
        h = h + F.silu(self.linear6(h))
        h = h + F.silu(self.linear7(h))
        return self.linear8(h)

    def forward(self, x):
        """Classical auto-encoder

        Args:
            x (float): state

        Returns:
            float: state via auto-encoder
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
    
class AutoEncoderMLP(nn.Module):
    """It is meant to have nonlinear encoder and decoder. Precisely, here we have hardcoded three hidden layers.
    It is inhreited from nn.Module.
    """

    def __init__(self, dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, latent_dim)

        self.linear5 = torch.nn.Linear(latent_dim, hidden_dim)
        self.linear6 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = torch.nn.Linear(hidden_dim, dim)

    def encode(self, x):
        """Encoder function

        Args:
            x (float): state

        Returns:
            float: encoded state
        """
        h = self.linear1(x)
        h = h + F.silu(self.linear2(h))
        h = h + F.silu(self.linear3(h))
        return self.linear4(h)

    def decode(self, z):
        """Decoder function

        Args:
            z (float): encoded state

        Returns:
            float: decoded state
        """
        h = self.linear5(z)
        h = h + F.silu(self.linear6(h))
        h = h + F.silu(self.linear7(h))
        return self.linear8(h)

    def forward(self, x):
        """Classical auto-encoder

        Args:
            x (float): state

        Returns:
            float: state via auto-encoder
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class ModelHypothesisGlobalStable(nn.Module):
    """
        Define model parameters.
        A = (J-R)Q,
        where J is skew-symmetric, R and Q are positive semi-definite matrics.
        We have assumed Q to be identity; hence, it is removed from the optimzer.
        This then guarantees stability of A.
    """
    def __init__(self, dim, *arg, B_term=False,  **kwargs):
        """
            Define model parameters.
            A = (J-R)Q,
            where J is skew-symmetric, R and Q are positive semi-definite matrics.
            We have assumed Q to be identity; hence, it is removed from the optimzer.
            This then guarantees stability of A.
        """
        super().__init__()
        print("Global Stability Gurantees!")
        FAC = 10
        self.B_term = B_term
        sys_order = dim
        self.sys_order = sys_order
        self._J = torch.nn.Parameter(torch.randn(sys_order, sys_order) / FAC)
        self._R = torch.nn.Parameter(torch.randn(sys_order, sys_order) / FAC)

        self.B = torch.nn.Parameter(torch.zeros(sys_order, 1))
        self._H_tensor = torch.nn.Parameter(
            torch.zeros(sys_order, sys_order, sys_order) / FAC
        )
        print("B_term:", self.B_term)

    @property
    def A(self):
        J = self._J - self._J.T
        R = self._R @ self._R.T
        _A = J - R

        self._A = _A
        return self._A

    @property
    def H(self):
        _H_tensor2 = self._H_tensor.permute(0, 2, 1)
        J_tensor = self._H_tensor - _H_tensor2
        self._H = J_tensor.permute(1, 0, 2).reshape(self.sys_order, self.sys_order**2)
        return self._H

    def forward(self, x, t):
        x_A = x @ self.A.T
        x_H = kron(x, x) @ self.H.T
        if self.B_term:
            return x_A + x_H + self.B.T
        else:
            return x_A + x_H

    def vector_field(self, x):
        """it defines vector field

        Args:
            x (float): state

        Returns:
            float: vector field
        """
        return self.forward(x, t=0)


class ModelHypothesisQuad(nn.Module):
    """
        Define quad model parameters.
    """
    def __init__(self, dim, *arg, B_term=True,  **kwargs):
        """
            Define model parameters.
            A = (J-R)Q,
            where J is skew-symmetric, R and Q are positive semi-definite matrics.
            We have assumed Q to be identity; hence, it is removed from the optimzer.
            This then guarantees stability of A.
        """
        super().__init__()
        print("No Stability Gurantees!")
        FAC = 10
        self.B_term = B_term
        sys_order = dim
        self.sys_order = sys_order
        self._A = torch.nn.Parameter(torch.randn(sys_order, sys_order) / FAC)
        

        self.B = torch.nn.Parameter(torch.zeros(sys_order, 1))
        self._H_tensor = torch.nn.Parameter(
            torch.zeros(sys_order, sys_order, sys_order) / FAC
        )
        print("B_term:", self.B_term)

    @property
    def A(self):
        return self._A

    @property
    def H(self):
        self._H = self._H_tensor.reshape(self.sys_order, self.sys_order**2)
        return self._H

    def forward(self, x, t):
        x_A = x @ self.A.T
        x_H = kron(x, x) @ self.H.T
        if self.B_term:
            return x_A + x_H + self.B.T
        else:
            return x_A + x_H

    def vector_field(self, x):
        """it defines vector field

        Args:
            x (float): state

        Returns:
            float: vector field
        """
        return self.forward(x, t=0)


class ModelHypothesisLinear(nn.Module):
    """

    """
    def __init__(self, dim, *arg, B_term=False,  **kwargs):
        """

        """
        super().__init__()
        print("No Stability Gurantees!")
        FAC = 10
        self.B_term = B_term
        sys_order = dim
        self.sys_order = sys_order
        self._J = torch.nn.Parameter(torch.randn(sys_order, sys_order) / FAC)

        self.B = torch.nn.Parameter(torch.zeros(sys_order, 1))

        print("B_term:", self.B_term)

    @property
    def A(self):
        J = self._J
        self._A = J
        return self._A

    def forward(self, x, t):
        x_A = x @ self.A.T
        if self.B_term:
            return x_A + self.B.T
        else:
            return x_A

    def vector_field(self, x):
        """it defines vector field

        Args:
            x (float): state

        Returns:
            float: vector field
        """
        return self.forward(x, t=0)



def kron(x, y):
    return torch.einsum("ab,ad->abd", [x, y]).view(x.size(0), x.size(1) * y.size(1))


def network_models(params):
    """It configures type of autoencoder and Hamiltonian function based on a given configuation.

    Args:
        params (dataclass): contains parameters

    Raises:
        ValueError: type of configuration is not found in setting, then it raises an error

    Returns:
        models: models containing autoencoder and Hamiltonian function
    """
    AE_VF_CONFIG = {
        "linear": (
            AutoEncoderMLP,
            ModelHypothesisLinear,
            "Nonlinear autoencoder and linear system with gaurantee stability!",
        ),
        # "linear_nostability": (
        #     AutoEncoderMLP,
        #     ModelHypothesisLinear,
        #     "Nonlinear autoencoder and linear system with NO-gaurantee stability!",
        # ),
        "quad": (
            AutoEncoderMLP,
            ModelHypothesisGlobalStable,
            "Nonlinear autoencoder and quadratic system with Gaurantee stability!",
        ),
        # "cubic": (
        #     AutoEncoderMLP,
        #     ModelHypothesisGlobalStable,
        #     "Nonlinear autoencoder and cubic system with gaurantee stability!",
        # ),
        "quad_opinf": (
            AutoEncoderLinear,
            ModelHypothesisQuad,
            "Linear autoencoder and quad system with NO-gaurantee stability!",
        ),
    }

    if params.confi_model in AE_VF_CONFIG:
        ae_fun, vf_fuc, print_str = AE_VF_CONFIG[params.confi_model]
        print(print_str)
        models = {
            "vf": vf_fuc(dim=params.latent_dim).double().to(device),
            "ae": ae_fun(
                dim=params.canonical_dim,
                hidden_dim=params.hidden_dim,
                latent_dim=params.latent_dim,
            )
            .double()
            .to(device),
        }

    else:
        raise ValueError(
            f" '{params.confi_model}' configuration is not found! Kindly provide suitable model configuration."
        )

    return models


def train(models, train_dl, optim, params):
    """It does training to learn vector field for the systems that are canonical

    Args:
        models (nn.module): models containing autoencoder and Hamiltonian networks
        train_dl (dataloder): training data
        optim (optimizer): optmizer to update parameters of neural networks
        params (dataclass): contains necessary parameter e.g., number of epochs

    Returns:
        (model, loss): trained model and loss as training progresses
    """
    models["ae"].train()
    models["hnn"].train()

    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=1500 * len(train_dl), gamma=0.1
    )

    print("Training begins!")

    mse_loss = nn.MSELoss()

    err_t = []
    for i in range(params.epoch):

        for x, dxdt in train_dl:
            z = models["ae"].encode(x)
            x_hat = models["ae"].decode(z)
            loss_ae = 0.5 * mse_loss(x_hat, x)  # Encoder loss
            loss_ae += 0.5 * (x_hat - x).abs().mean()  # Encoder loss


            dzdx = util.jacobian_bcw(z, x)

            dzdt = batch_mtxproduct(dzdx, dxdt)
            dzdt_hat = models["hnn"].vector_field(z)
            # Hamiltonian neural network loss
            loss_h = mse_loss(dzdt_hat, dzdt)

            ### Extra constraints (symplectic transformation)
            J = util.tensor_J(len(x), params.latent_dim).to(device)
            J_can = util.tensor_J(len(x), params.canonical_dim).to(device)

            dzdxT = torch.transpose(dzdx, 1, 2)
            Can = util.can_tp(dzdxT, J)
            Can = util.can_tp(Can, dzdx)
            loss_can = mse_loss(Can, J_can)

            ###########
            if params.confi_model in {"linear", "linear_opinf"}:
                loss = 1e-1 * loss_ae + loss_h + loss_can
            elif params.confi_model == "linear_nostability":
                loss = 1e-1 * loss_ae + loss_h
            else:
                loss = (
                    1e-0 * loss_ae
                    + loss_h
                    + loss_can
                    + 1e-4 * (models["hnn"].fc_quad.weight).abs().mean()
                )
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            err_t.append(loss.item())

        if (i + 1) % 200 == 0:
            lr = optim.param_groups[0]["lr"]
            print(
                f"Epoch {i+1}/{params.epoch} | loss_HNN: {loss_h.item():.2e}| loss_can: {loss_can.item():.2e} | loss_AE: {loss_ae.item():.2e} | learning rate: {lr:.2e}"
            )
    return models, err_t


def train_quad(models, train_dl, optim, params):
    """It does training to learn vector field for the systems that are canonical

    Args:
        models (nn.module): models containing autoencoder and Hamiltonian networks
        train_dl (dataloder): training data
        optim (optimizer): optmizer to update parameters of neural networks
        params (dataclass): contains necessary parameter e.g., number of epochs

    Returns:
        (model, loss): trained model and loss as training progresses
    """
    models["ae"].train()
    models["vf"].train()

    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size= int(3*params.epoch/8)* len(train_dl), gamma=0.1
    )

    print("Training begins!")

    mse_loss = nn.MSELoss()

    err_t = []
    loss_weights = params.loss_weights
    print(f'weights for loss function are: {loss_weights}')
    for i in range(params.epoch):

        for x, dxdt in train_dl:

            z = models["ae"].encode(x)
            x_hat = models["ae"].decode(z)
            loss_ae = 0.5 * mse_loss(x_hat, x)  # Encoder loss
            loss_ae += 0.5 * (x_hat - x).abs().mean()  # Encoder loss

            dzdx = util.jacobian_bcw(z, x)
            dzdt = batch_mtxproduct(dzdx, dxdt)
            # print(dzdx.shape, dxdt.shape, dzdt.shape)

            dzdt_hat = models["vf"].vector_field(z)
            # Hamiltonian neural network loss
            loss_vf = 0.5*mse_loss(dzdt_hat, dzdt)
            loss_vf += 0.5*(dzdt_hat - dzdt).abs().mean()

            
#             dxdz = util.jacobian_bcw(x_hat, z)
#             # # # print(dxdz.shape, dzdt.shape)
#             dxdt_estimate = batch_mtxproduct(dxdz, dzdt)
#             # # print(dxdt_estimate.shape, dxdt.shape)
#             loss_vf += 0.5*mse_loss(dxdt_estimate, dxdt)
#             loss_vf += 0.5*((dxdt_estimate - dxdt).abs().mean())

            ###########
            if params.confi_model in {"linear", "linear_opinf"}:
                loss = loss_weights[0] * loss_ae + loss_weights[1] * loss_vf
            elif params.confi_model == "linear_nostability":
                loss = loss_weights[0] * loss_ae + loss_weights[1] * loss_vf
            else:
                loss = (
                    loss_weights[0] * loss_ae
                    + loss_weights[1]* loss_vf
                    + 1e-5 * (models["vf"]._H_tensor).abs().mean()
                )
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            err_t.append(loss.item())

        if (i + 1) % 1 == 0:
            lr = optim.param_groups[0]["lr"]
            print(
                f"Epoch {i+1}/{params.epoch} | loss_VF: {loss_vf.item():.2e} | loss_AE: {loss_ae.item():.2e} | learning rate: {lr:.2e}"
            )
    return models, err_t


def train_quad_conv(models, train_dl, optim, params):
    """It does training to learn vector field for the systems that are canonical

    Args:
        models (nn.module): models containing autoencoder and Hamiltonian networks
        train_dl (dataloder): training data
        optim (optimizer): optmizer to update parameters of neural networks
        params (dataclass): contains necessary parameter e.g., number of epochs

    Returns:
        (model, loss): trained model and loss as training progresses
    """
    print('function from train_quad_conv')
    models["ae"].train()
    models["vf"].train()

    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size= int(3*params.epoch/8)* len(train_dl), gamma=0.1
    )

    print("Training begins!")

    mse_loss = nn.MSELoss()

    err_t = []
    loss_weights = params.loss_weights
    print(f'weights for loss function are: {loss_weights}')
    for i in range(params.epoch):

        for x, dxdt in train_dl:

            z = models["ae"].encode(x)
            x_hat = models["ae"].decode(z)
            loss_ae = 0.5 * mse_loss(x_hat, x)  # Encoder loss
            loss_ae += 0.5 * (x_hat - x).abs().mean()  # Encoder loss

            dzdx = util.jacobian_bcw(z, x)
            dzdt = batch_mtxproduct(dzdx, dxdt)
            # print(dzdx.shape, dxdt.shape, dzdt.shape)

            dzdt_hat = models["vf"].vector_field(z)
            # Hamiltonian neural network loss
            loss_vf = mse_loss(dzdt_hat, dzdt)
            # loss_vf += 0.5*(dzdt_hat - dzdt).abs().mean()

            
            # dxdz = util.jacobian_bcw(x_hat, z)
            # # print(dxdz.shape, dzdt.shape)
            # dxdt_estimate = batch_mtxproduct(dxdz, dzdt)
            # # print(dxdt_estimate.shape, dxdt.shape)
            # loss_vf += 0.5*mse_loss(dxdt_estimate, dxdt)




            ###########
            if params.confi_model in {"linear", "linear_opinf"}:
                loss = loss_weights[0] * loss_ae + loss_weights[1] * loss_vf
            elif params.confi_model == "linear_nostability":
                loss = loss_weights[0] * loss_ae + loss_weights[1] * loss_vf
            else:
                loss = (
                    loss_weights[0] * loss_ae
                    + loss_weights[1]* loss_vf
                    + 1e-5 * (models["vf"]._H_tensor).abs().mean()
                )
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            err_t.append(loss.item())

        if (i + 1) % 1 == 0:
            lr = optim.param_groups[0]["lr"]
            print(
                f"Epoch {i+1}/{params.epoch} | loss_VF: {loss_vf.item():.2e} | loss_AE: {loss_ae.item():.2e} | learning rate: {lr:.2e}"
            )
    return models, err_t



#############################################
### High-dimensional example ################
#############################################
class Decoder(torch.nn.Module):
    """Convolution-based decoder.
    It aims to construct full solutions using a convolution neural network.

    Args:
        latent_dim (int): latent space dimension
    """

    def __init__(self, latent_dim):
        super().__init__()

        # Decoder
        self.linear1 = nn.Linear(in_features=2 * latent_dim, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=512)
        self.deconv1 = nn.ConvTranspose1d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv2 = nn.ConvTranspose1d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv3 = nn.ConvTranspose1d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.deconv4 = nn.ConvTranspose1d(
            in_channels=8,
            out_channels=2,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)

    def decode(self, z):
        #     print('Decoding')
        z2 = z**2
        z_all = torch.cat((z, z2), dim=-1)

        h = self.linear1(z_all)
        h = F.selu_(h)
        h = self.linear2(h)
        h = F.selu_(h)
        h = h.reshape(h.shape[0], 64, 8)
        h = self.upsample(h)
        h = self.deconv1(h)
        h = F.selu_(h)
        h = self.deconv2(h)
        h = F.selu_(h)
        h = self.deconv3(h)
        h = F.selu_(h)
        h = self.deconv4(h)
        h = nn.Flatten()(h)
        return h

    def forward(self, z):
        return self.decode(z)


class DecoderQuad(nn.Module):
    """Quadratic decoder. It aims to construct full solutions using a quadratic ansatz.

    Args:
        latent_dim (int): latent space dimension
    """

    def __init__(self, latent_dim):
        super().__init__()

        # Decoder
        self.linear = nn.Linear(
            in_features=latent_dim + latent_dim**2, out_features=512
        )

    def decode(self, z):
        z2 = util.kron(z, z)
        z_all = torch.cat((z, z2), dim=-1)

        h = self.linear(z_all)
        return h

    def forward(self, z):
        return self.decode(z)


def train_decoder(decoder, optim, train_dl, params, V_proj):
    """Training of decoder that aims to reconstruct spatial solutions using OD coordindates.

    Args:
        decoder (nn.Module): decoder
        optim (optimizer): optimizer
        train_dl (dataloader): training data
        params (dataclass): contains necessary parameters for training
        V_proj (float): POD projection matrix

    Returns:
        decoder, err_t: trained decoder and error as training progresses
    """

    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=250 * len(train_dl), gamma=0.1
    )

    print("training started")
    mse_loss = nn.MSELoss()

    err_t = []
    for i in range(params.epoch):
        for x, z in train_dl:
            x_hat = decoder(z)  # reconstion using decoder
            x_pod = z @ V_proj.T  # reconstruction using POD matrix
            loss_ae = 0.5 * mse_loss(x_hat, x)  # Encoder loss
            loss_ae += 0.5 * (x_hat - x).abs().mean()

            with torch.no_grad():
                loss_pod = 0.5 * mse_loss(x_pod, x)  # Encoder loss
                loss_pod += 0.5 * (x_pod - x).abs().mean()

            loss = loss_ae
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            err_t.append(loss.item())

        if (i + 1) % 100 == 0:
            lr = optim.param_groups[0]["lr"]
            print(
                f"Epoch {i+1}/{params.epoch} | loss_AE: {loss_ae.item():.2e} | loss_POD: {loss_pod.item():.2e} | learning rate: {lr:.2e}"
            )
    return decoder, err_t
