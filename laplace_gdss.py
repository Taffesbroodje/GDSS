"""
Last-Layer Laplace Approximation for GDSS (Graph Diffusion via SDEs).

Adapted from DIFF-UQ (Jazbec et al.) for graph generation with score-based models.
This module provides uncertainty quantification for the score networks in GDSS.
"""

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset

from laplace.curvature.curvlinops import CurvlinopsEF
from laplace.baselaplace import DiagLaplace


class GDSSCurvlinopsEF(CurvlinopsEF):
    """
    Custom curvature backend for GDSS score networks.
    Computes diagonal empirical Fisher for graph-structured inputs.
    """

    def gradients(self, x, adj, flags, t, target, model_type='x'):
        """
        Compute batch gradients for GDSS score network.

        Parameters
        ----------
        x : torch.Tensor
            Node features (batch, num_nodes, feat_dim)
        adj : torch.Tensor
            Adjacency matrix (batch, num_nodes, num_nodes)
        flags : torch.Tensor
            Node flags indicating valid nodes
        t : torch.Tensor
            Diffusion timestep (batch,)
        target : torch.Tensor
            Target score (the noise z)
        model_type : str
            'x' for ScoreNetworkX, 'adj' for ScoreNetworkA

        Returns
        -------
        Gs : torch.Tensor
            Gradients (batch, num_params)
        loss : torch.Tensor
            Batch loss
        """
        def loss_single(x_i, adj_i, flags_i, t_i, target_i, params_dict, buffers_dict):
            x_i = x_i.unsqueeze(0)
            adj_i = adj_i.unsqueeze(0)
            flags_i = flags_i.unsqueeze(0)

            output = torch.func.functional_call(
                self.model,
                (params_dict, buffers_dict),
                (x_i, adj_i, flags_i)
            )

            # Score matching loss: ||score * std + z||^2
            # Here we just use MSE between output and target for Fisher computation
            loss = torch.func.functional_call(
                self.lossfunc, {}, (output.flatten(), target_i.flatten())
            )
            return loss, loss

        grad_fn = torch.func.grad(loss_single, argnums=5, has_aux=True)
        batch_grad_fn = torch.func.vmap(grad_fn, in_dims=(0, 0, 0, 0, 0, None, None))

        batch_grad, batch_loss = batch_grad_fn(
            x, adj, flags, t, target, self.params_dict, self.buffers_dict
        )
        Gs = torch.cat([bg.flatten(start_dim=1) for bg in batch_grad.values()], dim=1)

        if self.subnetwork_indices is not None:
            Gs = Gs[:, self.subnetwork_indices]

        loss = batch_loss.sum(0)
        return Gs, loss

    def diag(self, x, adj, flags, t, target, model_type='x', **kwargs):
        Gs, loss = self.gradients(x, adj, flags, t, target, model_type)
        Gs, loss = Gs.detach(), loss.detach()
        diag_ef = torch.einsum("bp,bp->p", Gs, Gs)
        return self.factor * loss, self.factor * diag_ef


class GDSSLaplace(DiagLaplace):
    """
    Last-Layer Diagonal Laplace Approximation for GDSS score networks.

    This class wraps a GDSS score network (ScoreNetworkX or ScoreNetworkA)
    and fits a Laplace approximation on the last layer for uncertainty quantification.
    """

    def __init__(
        self,
        model,
        model_type='x',  # 'x' for ScoreNetworkX, 'adj' for ScoreNetworkA
        last_layer_name='final.linears.2',  # Last linear layer in the MLP
        likelihood="regression",
        sigma_noise=1.0,
        prior_precision=1.0,
        prior_mean=0.0,
        temperature=1.0,
    ):
        """
        Initialize the GDSS Laplace wrapper.

        Parameters
        ----------
        model : nn.Module
            GDSS score network (ScoreNetworkX or ScoreNetworkA)
        model_type : str
            'x' for node feature network, 'adj' for adjacency network
        last_layer_name : str
            Name of the last layer to apply Laplace to
        """
        self.model_type = model_type
        self.last_layer_name = last_layer_name

        # Count and freeze parameters
        sum_param = 0
        sum_param_grad = 0
        sum_param_final_layer = 0

        for name, param in model.named_parameters():
            sum_param += param.numel()
            if param.requires_grad:
                sum_param_grad += param.numel()
            if last_layer_name in name:
                sum_param_final_layer += param.numel()
            if last_layer_name not in name:
                param.requires_grad = False

        print(f"[GDSSLaplace] Model type: {model_type}")
        print(f"[GDSSLaplace] Total parameters: {sum_param}")
        print(f"[GDSSLaplace] Parameters with grad before: {sum_param_grad}")
        print(f"[GDSSLaplace] Last layer parameters: {sum_param_final_layer}")

        # Initialize parent class with custom backend
        super().__init__(
            model,
            likelihood,
            sigma_noise,
            prior_precision,
            prior_mean,
            temperature,
            backend=GDSSCurvlinopsEF
        )

    def fit(self, train_loader, sde, override=True):
        """
        Fit the Laplace approximation using training data.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader yielding (x, adj) batches
        sde : SDE
            The SDE object for computing noise scales
        override : bool
            Whether to reset the Hessian approximation
        """
        if override:
            self._init_H()
            self.loss = 0
            self.n_data = 0

        self.model.eval()
        self.mean = parameters_to_vector(self.params)
        if not self.enable_backprop:
            self.mean = self.mean.detach()

        N = len(train_loader.dataset)
        device = next(self.model.parameters()).device
        eps = 1e-5

        for batch_idx, (x, adj) in enumerate(train_loader):
            x, adj = x.to(device), adj.to(device)

            # Get node flags
            flags = (adj.sum(-1) > 0).float()

            # Sample random timestep
            t = torch.rand(adj.shape[0], device=device) * (sde.T - eps) + eps

            # Generate noise (target for score matching)
            if self.model_type == 'x':
                z = torch.randn_like(x)
                # Mask noise
                z = z * flags.unsqueeze(-1)
                target = z
            else:  # adj
                z = torch.randn_like(adj)
                # Make symmetric and mask
                z = (z + z.transpose(-1, -2)) / 2
                z = z * flags.unsqueeze(-1) * flags.unsqueeze(-2)
                target = z

            # Perturb data
            mean, std = sde.marginal_prob(x if self.model_type == 'x' else adj, t)

            self.model.zero_grad()
            loss_batch, H_batch = self.backend.diag(
                x, adj, flags, t, target, self.model_type, N=N
            )
            self.loss += loss_batch
            self.H += H_batch

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}")

        self.n_data += N
        print(f"[GDSSLaplace] Fit complete. N={N}, loss={self.loss:.4f}")

    @torch.no_grad()
    def predictive_samples(self, x, adj, flags, n_samples=100):
        """
        Generate predictive samples from the Laplace posterior.

        Parameters
        ----------
        x : torch.Tensor
            Node features
        adj : torch.Tensor
            Adjacency matrix
        flags : torch.Tensor
            Node flags
        n_samples : int
            Number of posterior samples

        Returns
        -------
        samples : torch.Tensor
            Score predictions from posterior samples (n_samples, batch, ...)
        """
        self.model.eval()

        # Get posterior std
        posterior_precision = self.H + self.prior_precision
        posterior_var = 1.0 / (posterior_precision + 1e-6)
        posterior_std = torch.sqrt(posterior_var)

        samples = []
        original_params = parameters_to_vector(self.params).clone()

        for _ in range(n_samples):
            # Sample parameters from posterior
            eps = torch.randn_like(self.mean)
            sampled_params = self.mean + posterior_std * eps

            # Set sampled parameters
            vector_to_parameters(sampled_params, self.params)

            # Forward pass
            output = self.model(x, adj, flags)
            samples.append(output.clone())

        # Restore original parameters
        vector_to_parameters(original_params, self.params)

        return torch.stack(samples, dim=0)

    @torch.no_grad()
    def get_uncertainty(self, x, adj, flags, n_samples=50):
        """
        Get uncertainty estimate for score predictions.

        Returns mean prediction and standard deviation across posterior samples.
        """
        samples = self.predictive_samples(x, adj, flags, n_samples)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return mean, std


def create_laplace_wrapper(model, model_type='x', sde=None, train_loader=None,
                           prior_precision=1.0, fit=True):
    """
    Convenience function to create and optionally fit a GDSS Laplace wrapper.

    Parameters
    ----------
    model : nn.Module
        GDSS score network
    model_type : str
        'x' or 'adj'
    sde : SDE
        SDE object for noise computation
    train_loader : DataLoader
        Training data loader
    prior_precision : float
        Prior precision (regularization)
    fit : bool
        Whether to fit immediately

    Returns
    -------
    laplace : GDSSLaplace
        Fitted Laplace wrapper
    """
    laplace = GDSSLaplace(
        model=model,
        model_type=model_type,
        prior_precision=prior_precision,
    )

    if fit and train_loader is not None and sde is not None:
        laplace.fit(train_loader, sde)

    return laplace
