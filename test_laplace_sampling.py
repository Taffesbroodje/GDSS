"""
Test Laplace uncertainty estimation during GDSS sampling.
Uses pretrained checkpoint to generate graphs with uncertainty.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, '.')

from models.ScoreNetwork_X import ScoreNetworkX
from models.ScoreNetwork_A import ScoreNetworkA
from sde import VPSDE
from laplace_gdss import GDSSLaplace
from utils.graph_utils import mask_x, mask_adjs, gen_noise


def load_models_from_checkpoint(ckpt_path, device):
    """Load GDSS models from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    params_x = ckpt['params_x'].copy()
    params_adj = ckpt['params_adj'].copy()

    # Remove model_type key (used for dispatching, not for constructor)
    model_type_x = params_x.pop('model_type', 'ScoreNetworkX')
    model_type_adj = params_adj.pop('model_type', 'ScoreNetworkA')

    # Create models based on params
    if model_type_x == 'ScoreNetworkX_GMH' or 'num_heads' in params_x:
        from models.ScoreNetwork_X import ScoreNetworkX_GMH
        model_x = ScoreNetworkX_GMH(**params_x).to(device)
    else:
        model_x = ScoreNetworkX(**params_x).to(device)

    model_adj = ScoreNetworkA(**params_adj).to(device)

    # Load state dicts
    model_x.load_state_dict(ckpt['x_state_dict'])
    model_adj.load_state_dict(ckpt['adj_state_dict'])

    return model_x, model_adj, ckpt['model_config']


def simple_sampling_with_uncertainty(
    model_x, model_adj,
    laplace_x, laplace_adj,
    sde_x, sde_adj,
    shape_x, shape_adj,
    n_steps=50,
    n_uncertainty_samples=20,
    device='cpu'
):
    """
    Simple sampling loop with uncertainty estimation.

    Returns generated samples and uncertainty estimates at each step.
    """
    model_x.eval()
    model_adj.eval()

    eps = 1e-3
    batch_size = shape_x[0]
    num_nodes = shape_x[1]

    # Initial samples from prior
    x = torch.randn(shape_x, device=device)
    adj = torch.randn(shape_adj, device=device)
    adj = (adj + adj.transpose(-1, -2)) / 2  # Symmetrize

    # Flags (all nodes valid for simplicity)
    flags = torch.ones(batch_size, num_nodes, device=device)

    x = mask_x(x, flags)
    adj = mask_adjs(adj, flags)

    timesteps = torch.linspace(sde_x.T, eps, n_steps, device=device)
    dt = -1.0 / n_steps

    uncertainties_x = []
    uncertainties_adj = []

    print(f"\nSampling {batch_size} graphs with {n_steps} steps...")

    for i, t in enumerate(timesteps):
        vec_t = torch.ones(batch_size, device=device) * t

        with torch.no_grad():
            # Get score predictions
            score_x = model_x(x, adj, flags)
            score_adj = model_adj(x, adj, flags)

            # Get uncertainty estimates
            _, std_x = laplace_x.get_uncertainty(x, adj, flags, n_samples=n_uncertainty_samples)
            _, std_adj = laplace_adj.get_uncertainty(x, adj, flags, n_samples=n_uncertainty_samples)

            uncertainties_x.append(std_x.mean().item())
            uncertainties_adj.append(std_adj.mean().item())

            # Simple Euler-Maruyama update
            # drift = -0.5 * beta(t) * x - beta(t) * score
            # For simplicity, just use score-based update
            _, std = sde_x.marginal_prob(torch.zeros_like(x), vec_t)
            score_x = -score_x / std[:, None, None]

            _, std = sde_adj.marginal_prob(torch.zeros_like(adj), vec_t)
            score_adj = -score_adj / std[:, None, None]

            # Update
            noise_x = torch.randn_like(x) * np.sqrt(-dt)
            noise_adj = torch.randn_like(adj) * np.sqrt(-dt)
            noise_adj = (noise_adj + noise_adj.transpose(-1, -2)) / 2

            x = x + score_x * dt + noise_x * 0.1
            adj = adj + score_adj * dt + noise_adj * 0.1

            x = mask_x(x, flags)
            adj = mask_adjs(adj, flags)

        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/{n_steps}: unc_x={uncertainties_x[-1]:.4f}, unc_adj={uncertainties_adj[-1]:.4f}")

    return x, adj, uncertainties_x, uncertainties_adj


def main():
    print("="*60)
    print("GDSS Sampling with Laplace Uncertainty")
    print("="*60)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Check for checkpoint
    import os
    ckpt_path = "./checkpoints/community_small/gdss_community_small.pth"

    if not os.path.exists(ckpt_path):
        print(f"\nCheckpoint not found: {ckpt_path}")
        print("Running with random models instead...\n")

        # Create random small models for testing
        model_x = ScoreNetworkX(
            max_feat_num=2,
            depth=2,
            nhid=32
        ).to(device)

        model_adj = ScoreNetworkA(
            max_feat_num=2,
            max_node_num=12,
            nhid=32,
            num_layers=2,
            num_linears=2,
            c_init=2,
            c_hid=4,
            c_final=2,
            adim=16,
            num_heads=2,
            conv='GCN'
        ).to(device)

        shape_x = (4, 12, 2)  # batch, nodes, features
        shape_adj = (4, 12, 12)  # batch, nodes, nodes
    else:
        print(f"\nLoading checkpoint: {ckpt_path}")
        model_x, model_adj, config = load_models_from_checkpoint(ckpt_path, device)

        print(f"Config: {config.data.data}")
        print(f"Max nodes: {config.data.max_node_num}")
        print(f"Max features: {config.data.max_feat_num}")

        shape_x = (4, config.data.max_node_num, config.data.max_feat_num)
        shape_adj = (4, config.data.max_node_num, config.data.max_node_num)

    # Create SDEs
    sde_x = VPSDE(beta_min=0.1, beta_max=1.0, N=100)
    sde_adj = VPSDE(beta_min=0.1, beta_max=1.0, N=100)

    # Create Laplace wrappers
    print("\nCreating Laplace wrappers...")
    laplace_x = GDSSLaplace(
        model=model_x,
        model_type='x',
        last_layer_name='final.linears.2',
        prior_precision=1.0
    )

    laplace_adj = GDSSLaplace(
        model=model_adj,
        model_type='adj',
        last_layer_name='final.linears.2',
        prior_precision=1.0
    )

    # Run sampling with uncertainty
    x_gen, adj_gen, unc_x, unc_adj = simple_sampling_with_uncertainty(
        model_x, model_adj,
        laplace_x, laplace_adj,
        sde_x, sde_adj,
        shape_x, shape_adj,
        n_steps=50,
        n_uncertainty_samples=10,
        device=device
    )

    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"Generated x shape: {x_gen.shape}")
    print(f"Generated adj shape: {adj_gen.shape}")
    print(f"X value range: [{x_gen.min():.3f}, {x_gen.max():.3f}]")
    print(f"Adj value range: [{adj_gen.min():.3f}, {adj_gen.max():.3f}]")

    print(f"\nUncertainty evolution:")
    print(f"  X uncertainty: {unc_x[0]:.4f} -> {unc_x[-1]:.4f}")
    print(f"  Adj uncertainty: {unc_adj[0]:.4f} -> {unc_adj[-1]:.4f}")

    # Simple visualization of uncertainty over time
    print("\nUncertainty over sampling steps (X):")
    for i in range(0, len(unc_x), 10):
        bar = "█" * int(unc_x[i] * 10)
        print(f"  Step {i:3d}: {bar} {unc_x[i]:.4f}")

    print("\n✓ Sampling with uncertainty estimation complete!")
    return True


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
