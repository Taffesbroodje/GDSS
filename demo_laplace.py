"""
Demo: Last-Layer Laplace Approximation for GDSS

This demo shows how to:
1. Load a pretrained GDSS model
2. Wrap it with Laplace approximation
3. Get uncertainty estimates for score predictions
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from models.ScoreNetwork_X import ScoreNetworkX
from models.ScoreNetwork_A import ScoreNetworkA
from sde import VPSDE
from laplace_gdss import GDSSLaplace
from utils.graph_utils import mask_x, mask_adjs


def load_model_from_checkpoint(ckpt_path, device):
    """Load GDSS models from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    params_x = ckpt['params_x'].copy()
    params_adj = ckpt['params_adj'].copy()

    # Remove model_type key
    params_x.pop('model_type', None)
    params_adj.pop('model_type', None)

    model_x = ScoreNetworkX(**params_x).to(device)
    model_adj = ScoreNetworkA(**params_adj).to(device)

    model_x.load_state_dict(ckpt['x_state_dict'])
    model_adj.load_state_dict(ckpt['adj_state_dict'])

    return model_x, model_adj, ckpt['model_config']


def main():
    print("="*60)
    print("GDSS Last-Layer Laplace Demo")
    print("="*60)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}\n")

    # Load pretrained model
    ckpt_path = "./checkpoints/community_small/gdss_community_small.pth"
    print(f"Loading checkpoint: {ckpt_path}")

    model_x, model_adj, config = load_model_from_checkpoint(ckpt_path, device)

    print(f"Dataset: {config.data.data}")
    print(f"Max nodes: {config.data.max_node_num}")
    print(f"Max features: {config.data.max_feat_num}\n")

    # Create Laplace wrappers
    print("Creating Laplace wrappers (last layer only)...")
    laplace_x = GDSSLaplace(
        model=model_x,
        model_type='x',
        last_layer_name='final.linears.2',
        prior_precision=100.0,  # Higher prior = less uncertainty
    )

    laplace_adj = GDSSLaplace(
        model=model_adj,
        model_type='adj',
        last_layer_name='final.linears.2',
        prior_precision=100.0,
    )

    # Create test input
    print("\n" + "-"*60)
    print("Testing uncertainty estimation on random input")
    print("-"*60)

    batch_size = 4
    num_nodes = config.data.max_node_num
    feat_dim = config.data.max_feat_num

    # Random graph input
    x = torch.randn(batch_size, num_nodes, feat_dim, device=device)
    adj = torch.rand(batch_size, num_nodes, num_nodes, device=device)
    adj = (adj + adj.transpose(-1, -2)) / 2  # Symmetrize
    flags = torch.ones(batch_size, num_nodes, device=device)

    x = mask_x(x, flags)
    adj = mask_adjs(adj, flags)

    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  adj: {adj.shape}")

    # Get predictions with uncertainty
    print("\nGetting predictions with uncertainty (20 posterior samples)...")

    with torch.no_grad():
        # Standard forward pass
        score_x = model_x(x, adj, flags)
        score_adj = model_adj(x, adj, flags)

        print(f"\nScore predictions:")
        print(f"  score_x shape: {score_x.shape}")
        print(f"  score_adj shape: {score_adj.shape}")

        # Get uncertainty via posterior sampling
        mean_x, std_x = laplace_x.get_uncertainty(x, adj, flags, n_samples=20)
        mean_adj, std_adj = laplace_adj.get_uncertainty(x, adj, flags, n_samples=20)

        print(f"\nUncertainty estimates:")
        print(f"  X score std: mean={std_x.mean():.4f}, min={std_x.min():.4f}, max={std_x.max():.4f}")
        print(f"  Adj score std: mean={std_adj.mean():.4f}, min={std_adj.min():.4f}, max={std_adj.max():.4f}")

    # Test with different noise levels
    print("\n" + "-"*60)
    print("Testing uncertainty vs noise level")
    print("-"*60)

    sde = VPSDE(beta_min=0.1, beta_max=1.0, N=100)

    print("\nUncertainty at different diffusion timesteps:")
    print("(Higher t = more noise = should have different uncertainty)")

    for t_val in [0.1, 0.5, 0.9]:
        t = torch.ones(batch_size, device=device) * t_val

        # Get noisy input at timestep t
        mean, std = sde.marginal_prob(x, t)
        z = torch.randn_like(x)
        x_noisy = mean + std[:, None, None] * z

        mean, std = sde.marginal_prob(adj, t)
        z = torch.randn_like(adj)
        z = (z + z.transpose(-1, -2)) / 2
        adj_noisy = mean + std[:, None, None] * z

        x_noisy = mask_x(x_noisy, flags)
        adj_noisy = mask_adjs(adj_noisy, flags)

        with torch.no_grad():
            _, std_x = laplace_x.get_uncertainty(x_noisy, adj_noisy, flags, n_samples=20)
            _, std_adj = laplace_adj.get_uncertainty(x_noisy, adj_noisy, flags, n_samples=20)

        print(f"  t={t_val:.1f}: std_x={std_x.mean():.4f}, std_adj={std_adj.mean():.4f}")

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
    print("""
Next steps for full implementation:
1. Fit the Laplace approximation on training data (compute Fisher)
2. Integrate uncertainty into the sampling loop
3. Use uncertainty for selective denoising or other applications

See laplace_gdss.py for the full implementation.
""")


if __name__ == "__main__":
    main()
