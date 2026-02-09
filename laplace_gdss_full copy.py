"""
Full Last-Layer Laplace Approximation for GDSS

This module provides:
1. Proper Fisher/Hessian fitting on training data
2. Uncertainty-aware sampling
3. Integration with GDSS's score-based diffusion

Prior: Standard isotropic Gaussian centered at MAP (trained weights)
Hessian: Diagonal Empirical Fisher approximation
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange

from sde import VPSDE, VESDE
from utils.graph_utils import mask_x, mask_adjs, gen_noise, node_flags


class GDSSLaplaceFull:
    """
    Full Last-Layer Laplace Approximation for GDSS score networks.

    This class:
    1. Freezes all parameters except the last linear layer
    2. Computes diagonal empirical Fisher on training data
    3. Provides uncertainty estimates via posterior sampling
    """

    def __init__(
        self,
        model,
        model_type='x',  # 'x' for ScoreNetworkX, 'adj' for ScoreNetworkA
        last_layer_name='final.linears.2',
        prior_precision=1.0,
        sigma_noise=1.0,
    ):
        """
        Initialize the Laplace wrapper.

        Parameters
        ----------
        model : nn.Module
            GDSS score network
        model_type : str
            'x' for node features, 'adj' for adjacency
        last_layer_name : str
            Name of the last layer (for freezing others)
        prior_precision : float
            Prior precision (higher = less uncertainty, more regularization)
        sigma_noise : float
            Observation noise (for likelihood)
        """
        self.model = model
        self.model_type = model_type
        self.last_layer_name = last_layer_name
        self.prior_precision = prior_precision
        self.sigma_noise = sigma_noise

        # Freeze all but last layer
        self.n_params = 0
        self.n_params_total = 0

        for name, param in model.named_parameters():
            self.n_params_total += param.numel()
            if last_layer_name in name:
                self.n_params += param.numel()
                param.requires_grad = True
            else:
                param.requires_grad = False

        print(f"[GDSSLaplace] Model type: {model_type}")
        print(f"[GDSSLaplace] Total parameters: {self.n_params_total}")
        print(f"[GDSSLaplace] Last layer parameters: {self.n_params}")
        print(f"[GDSSLaplace] Prior precision: {prior_precision}")

        # Get last layer parameters
        self.last_layer_params = [p for p in model.parameters() if p.requires_grad]

        # Initialize Hessian (diagonal Fisher)
        self.H = None  # Will be (n_params,) tensor
        self.mean = None  # MAP estimate (trained weights)
        self.fitted = False

    def _get_param_vector(self):
        """Get flattened parameter vector for last layer."""
        return torch.cat([p.flatten() for p in self.last_layer_params])

    def _set_param_vector(self, vec):
        """Set last layer parameters from flattened vector."""
        idx = 0
        for p in self.last_layer_params:
            n = p.numel()
            p.data = vec[idx:idx+n].view_as(p)
            idx += n

    def fit(self, train_loader, sde, num_batches=None, device='cpu'):
        """
        Fit the Laplace approximation by computing diagonal Fisher.

        This computes H = Σ_i (∂L/∂θ)² over training data,
        where L is the score matching loss.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader yielding (x, adj) batches
        sde : SDE
            The SDE for computing noise scales
        num_batches : int, optional
            Number of batches to use (None = all)
        device : str or torch.device
            Device to use

        Returns
        -------
        fit_time : float
            Time taken to fit (seconds)
        """
        print(f"\n[GDSSLaplace] Fitting diagonal Fisher...")
        start_time = time.time()

        self.model.eval()
        device = next(self.model.parameters()).device

        # Store MAP estimate
        self.mean = self._get_param_vector().detach().clone()

        # Initialize diagonal Hessian
        self.H = torch.zeros(self.n_params, device=device)

        eps = 1e-5
        n_data = 0
        total_loss = 0

        n_batches = len(train_loader) if num_batches is None else min(num_batches, len(train_loader))

        pbar = tqdm(enumerate(train_loader), total=n_batches, desc="Fitting Fisher")

        for batch_idx, batch in pbar:
            if batch_idx >= n_batches:
                break

            if isinstance(batch, (list, tuple)):
                x, adj = batch[0], batch[1]
            else:
                x, adj = batch, batch

            x, adj = x.to(device), adj.to(device)
            batch_size = x.shape[0]
            n_data += batch_size

            # Get node flags
            flags = node_flags(adj)

            # Sample random timesteps
            t = torch.rand(batch_size, device=device) * (sde.T - eps) + eps

            # Generate noise (target)
            if self.model_type == 'x':
                z = gen_noise(x, flags, sym=False)
                mean_t, std_t = sde.marginal_prob(x, t)
                perturbed = mean_t + std_t[:, None, None] * z
                perturbed = mask_x(perturbed, flags)
            else:  # adj
                z = gen_noise(adj, flags, sym=True)
                mean_t, std_t = sde.marginal_prob(adj, t)
                perturbed = mean_t + std_t[:, None, None] * z
                perturbed = mask_adjs(perturbed, flags)

            # Compute gradients for each sample
            for i in range(batch_size):
                self.model.zero_grad()

                # Forward pass for single sample
                if self.model_type == 'x':
                    x_i = perturbed[i:i+1]
                    adj_i = adj[i:i+1]
                else:
                    x_i = x[i:i+1]
                    adj_i = perturbed[i:i+1]

                flags_i = flags[i:i+1]

                output = self.model(x_i, adj_i, flags_i)

                # Score matching loss: ||score + z/std||²
                # Simplified: MSE between output and -z/std
                std_i = std_t[i]
                target = -z[i:i+1] / std_i

                loss = torch.mean((output - target) ** 2)
                loss.backward()

                # Accumulate squared gradients (diagonal Fisher)
                grad_vec = torch.cat([p.grad.flatten() for p in self.last_layer_params])
                self.H += grad_vec ** 2

                total_loss += loss.item()

            pbar.set_postfix({'loss': total_loss / n_data})

        # Normalize by number of data points
        self.H = self.H / n_data

        self.fitted = True
        fit_time = time.time() - start_time

        print(f"[GDSSLaplace] Fitted on {n_data} samples in {fit_time:.1f}s")
        print(f"[GDSSLaplace] Fisher diag: min={self.H.min():.4f}, max={self.H.max():.4f}, mean={self.H.mean():.4f}")

        return fit_time

    def get_posterior_precision(self):
        """Get posterior precision (Fisher + prior)."""
        if not self.fitted:
            raise ValueError("Must call fit() first")
        return self.H + self.prior_precision

    def get_posterior_variance(self):
        """Get posterior variance."""
        precision = self.get_posterior_precision()
        return 1.0 / (precision + 1e-8)

    def get_posterior_std(self):
        """Get posterior standard deviation."""
        return torch.sqrt(self.get_posterior_variance())

    @torch.no_grad()
    def sample_parameters(self, n_samples=1):
        """
        Sample parameters from the Laplace posterior.

        Returns
        -------
        samples : list of torch.Tensor
            List of parameter vectors sampled from posterior
        """
        if not self.fitted:
            raise ValueError("Must call fit() first")

        std = self.get_posterior_std()
        samples = []

        for _ in range(n_samples):
            eps = torch.randn_like(self.mean)
            sample = self.mean + std * eps
            samples.append(sample)

        return samples

    @torch.no_grad()
    def predictive_samples(self, x, adj, flags, n_samples=20):
        """
        Get predictive samples from the posterior.

        Parameters
        ----------
        x, adj, flags : torch.Tensor
            Input graph
        n_samples : int
            Number of posterior samples

        Returns
        -------
        samples : torch.Tensor
            (n_samples, batch, ...) predictions
        """
        self.model.eval()

        original_params = self._get_param_vector().clone()
        param_samples = self.sample_parameters(n_samples)

        predictions = []
        for params in param_samples:
            self._set_param_vector(params)
            output = self.model(x, adj, flags)
            predictions.append(output.clone())

        # Restore original parameters
        self._set_param_vector(original_params)

        return torch.stack(predictions, dim=0)

    @torch.no_grad()
    def get_uncertainty(self, x, adj, flags, n_samples=20):
        """
        Get mean prediction and uncertainty.

        Returns
        -------
        mean : torch.Tensor
            Mean prediction
        std : torch.Tensor
            Standard deviation (uncertainty)
        """
        samples = self.predictive_samples(x, adj, flags, n_samples)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        return mean, std

    def save(self, path):
        """Save fitted Laplace state."""
        torch.save({
            'mean': self.mean,
            'H': self.H,
            'prior_precision': self.prior_precision,
            'sigma_noise': self.sigma_noise,
            'n_params': self.n_params,
            'model_type': self.model_type,
            'last_layer_name': self.last_layer_name,
            'fitted': self.fitted,
        }, path)
        print(f"[GDSSLaplace] Saved to {path}")

    def load(self, path):
        """Load fitted Laplace state."""
        state = torch.load(path, map_location=next(self.model.parameters()).device)
        self.mean = state['mean']
        self.H = state['H']
        self.prior_precision = state['prior_precision']
        self.sigma_noise = state['sigma_noise']
        self.fitted = state['fitted']
        print(f"[GDSSLaplace] Loaded from {path}")


def create_laplace_models(model_x, model_adj, prior_precision=1.0):
    """
    Create Laplace wrappers for both GDSS score networks.

    Returns
    -------
    laplace_x, laplace_adj : GDSSLaplaceFull
    """
    laplace_x = GDSSLaplaceFull(
        model=model_x,
        model_type='x',
        prior_precision=prior_precision,
    )

    laplace_adj = GDSSLaplaceFull(
        model=model_adj,
        model_type='adj',
        prior_precision=prior_precision,
    )

    return laplace_x, laplace_adj


def fit_laplace_models(laplace_x, laplace_adj, train_loader, sde_x, sde_adj,
                       num_batches=None, device='cpu'):
    """
    Fit both Laplace models on training data.

    Returns
    -------
    total_time : float
        Total fitting time in seconds
    """
    print("\n" + "="*60)
    print("Fitting Laplace approximation for ScoreNetworkX")
    print("="*60)
    time_x = laplace_x.fit(train_loader, sde_x, num_batches, device)

    print("\n" + "="*60)
    print("Fitting Laplace approximation for ScoreNetworkA")
    print("="*60)
    time_adj = laplace_adj.fit(train_loader, sde_adj, num_batches, device)

    return time_x + time_adj


# ============================================================
# Uncertainty-aware sampling
# ============================================================

def get_score_with_uncertainty(model, laplace, x, adj, flags, n_samples=10):
    """
    Get score prediction with uncertainty estimate.

    Returns
    -------
    score_mean : torch.Tensor
        Mean score prediction
    score_std : torch.Tensor
        Uncertainty (std) of score
    """
    if laplace.fitted:
        return laplace.get_uncertainty(x, adj, flags, n_samples)
    else:
        # Fall back to point estimate with zero uncertainty
        with torch.no_grad():
            score = model(x, adj, flags)
        return score, torch.zeros_like(score)


def uncertainty_aware_sampling(
    model_x, model_adj,
    laplace_x, laplace_adj,
    sde_x, sde_adj,
    init_flags,
    shape_x, shape_adj,
    n_steps=100,
    n_uncertainty_samples=10,
    eps=1e-3,
    device='cpu',
    return_trajectory=False,
):
    """
    Sample graphs with uncertainty tracking.

    This implements a simple Euler-Maruyama sampler that also
    computes uncertainty estimates at each step.

    Parameters
    ----------
    model_x, model_adj : nn.Module
        Score networks
    laplace_x, laplace_adj : GDSSLaplaceFull
        Fitted Laplace wrappers
    sde_x, sde_adj : SDE
        SDEs for x and adj
    init_flags : torch.Tensor
        Initial node flags
    shape_x, shape_adj : tuple
        Output shapes
    n_steps : int
        Number of diffusion steps
    n_uncertainty_samples : int
        Posterior samples for uncertainty
    eps : float
        Small epsilon for numerical stability
    device : str or torch.device
        Device
    return_trajectory : bool
        Whether to return full trajectory

    Returns
    -------
    x, adj : torch.Tensor
        Generated samples
    uncertainties : dict
        Dictionary with uncertainty trajectories
    """
    model_x.eval()
    model_adj.eval()

    # Initial samples from prior
    x = sde_x.prior_sampling(shape_x).to(device)
    adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
    flags = init_flags.to(device)

    x = mask_x(x, flags)
    adj = mask_adjs(adj, flags)

    timesteps = torch.linspace(sde_x.T, eps, n_steps, device=device)
    dt = -1.0 / n_steps

    # Track uncertainties
    unc_x_traj = []
    unc_adj_traj = []

    if return_trajectory:
        x_traj = [x.clone()]
        adj_traj = [adj.clone()]

    print(f"\nSampling with uncertainty ({n_steps} steps, {n_uncertainty_samples} posterior samples)...")

    for i in trange(n_steps, desc="Sampling"):
        t = timesteps[i]
        vec_t = torch.ones(shape_x[0], device=device) * t

        # Get scores with uncertainty
        score_x_mean, score_x_std = get_score_with_uncertainty(
            model_x, laplace_x, x, adj, flags, n_uncertainty_samples
        )
        score_adj_mean, score_adj_std = get_score_with_uncertainty(
            model_adj, laplace_adj, x, adj, flags, n_uncertainty_samples
        )

        # Track uncertainty
        unc_x_traj.append(score_x_std.mean().item())
        unc_adj_traj.append(score_adj_std.mean().item())

        # Convert to proper scores (divide by std)
        _, std_x = sde_x.marginal_prob(torch.zeros_like(x), vec_t)
        _, std_adj = sde_adj.marginal_prob(torch.zeros_like(adj), vec_t)

        score_x = -score_x_mean / std_x[:, None, None]
        score_adj = -score_adj_mean / std_adj[:, None, None]

        # Compute drift
        drift_x = -sde_x.sde(x, vec_t)[1][:, None, None] ** 2 * score_x
        drift_adj = -sde_adj.sde(adj, vec_t)[1][:, None, None] ** 2 * score_adj

        # Euler-Maruyama update
        noise_x = gen_noise(x, flags, sym=False) * np.sqrt(-dt)
        noise_adj = gen_noise(adj, flags, sym=True) * np.sqrt(-dt)

        diffusion_x = sde_x.sde(x, vec_t)[1][:, None, None]
        diffusion_adj = sde_adj.sde(adj, vec_t)[1][:, None, None]

        x = x + drift_x * dt + diffusion_x * noise_x
        adj = adj + drift_adj * dt + diffusion_adj * noise_adj

        x = mask_x(x, flags)
        adj = mask_adjs(adj, flags)

        if return_trajectory:
            x_traj.append(x.clone())
            adj_traj.append(adj.clone())

    uncertainties = {
        'x': unc_x_traj,
        'adj': unc_adj_traj,
    }

    if return_trajectory:
        uncertainties['x_traj'] = x_traj
        uncertainties['adj_traj'] = adj_traj

    return x, adj, uncertainties
