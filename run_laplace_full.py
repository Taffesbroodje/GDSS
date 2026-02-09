"""
Full Laplace Approximation Pipeline for GDSS

This script:
1. Loads pretrained GDSS models
2. Loads training data
3. Fits the diagonal Fisher (Laplace approximation)
4. Generates samples with uncertainty estimation

RUNTIME ESTIMATES (on Apple M-series):
- Fisher fitting: ~1-2 minutes (depends on num_batches)
- Sampling with uncertainty: ~2-3 minutes per batch

Usage:
    python run_laplace_full.py --dataset community_small --num_fit_batches 20
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, '.')

from models.ScoreNetwork_X import ScoreNetworkX
from models.ScoreNetwork_A import ScoreNetworkA
from sde import VPSDE
from utils.loader import load_data
from utils.graph_utils import init_flags, quantize, adjs_to_graphs
from laplace_gdss_full import (
    GDSSLaplaceFull,
    create_laplace_models,
    fit_laplace_models,
    uncertainty_aware_sampling,
    summarize_per_graph_uncertainty
)
from graph_semantic_uncertainty import graph_semantic_generative_uncertainty


def load_models_from_checkpoint(ckpt_path, device):
    """Load GDSS models from checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
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

    # Load EMA if available
    if 'ema_x' in ckpt:
        from utils.ema import ExponentialMovingAverage
        ema_x = ExponentialMovingAverage(model_x.parameters(), decay=0.9999)
        ema_x.load_state_dict(ckpt['ema_x'])
        ema_x.copy_to(model_x.parameters())

        ema_adj = ExponentialMovingAverage(model_adj.parameters(), decay=0.9999)
        ema_adj.load_state_dict(ckpt['ema_adj'])
        ema_adj.copy_to(model_adj.parameters())
        print("Loaded EMA weights")

    return model_x, model_adj, ckpt['model_config']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ENZYMES',
                        choices=['community_small', 'ego_small', 'grid', 'ENZYMES'])
    parser.add_argument('--num_fit_batches', type=int, default=20,
                        help='Number of batches for Fisher fitting (None=all)')
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of graphs to generate')
    parser.add_argument('--n_steps', type=int, default=100,
                        help='Number of diffusion steps')
    parser.add_argument('--n_uncertainty_samples', type=int, default=10,
                        help='Posterior samples for uncertainty')
    parser.add_argument('--prior_precision', type=float, default=1.0,
                        help='Prior precision (regularization)')
    parser.add_argument('--save_laplace', action='store_true',
                        help='Save fitted Laplace state')
    parser.add_argument('--load_laplace', type=str, default=None,
                        help='Load pre-fitted Laplace state')
    args = parser.parse_args()

    print("="*60)
    print("GDSS Full Laplace Pipeline")
    print("="*60)

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load checkpoint
    ckpt_path = f"./checkpoints/{args.dataset}/gdss_{args.dataset}.pth"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    model_x, model_adj, config = load_models_from_checkpoint(ckpt_path, device)

    print(f"\nDataset: {config.data.data}")
    print(f"Max nodes: {config.data.max_node_num}")
    print(f"Max features: {config.data.max_feat_num}")

    # Create SDEs
    sde_x = VPSDE(beta_min=0.1, beta_max=1.0, N=1000)
    sde_adj = VPSDE(beta_min=0.1, beta_max=1.0, N=1000)

    # Create Laplace wrappers
    print("\n" + "="*60)
    print("Creating Laplace Wrappers")
    print("="*60)
    laplace_x, laplace_adj = create_laplace_models(
        model_x, model_adj,
        prior_precision=args.prior_precision
    )

    # Load or fit Laplace
    if args.load_laplace and os.path.exists(args.load_laplace + '_x.pt'):
        print(f"\nLoading pre-fitted Laplace from {args.load_laplace}")
        laplace_x.load(args.load_laplace + '_x.pt')
        laplace_adj.load(args.load_laplace + '_adj.pt')
    else:
        # Load training data
        print("\n" + "="*60)
        print("Loading Training Data")
        print("="*60)
        train_loader, _ = load_data(config)
        print(f"Training batches: {len(train_loader)}")
        print(f"Batch size: {config.data.batch_size}")

        # Estimate time
        n_batches = args.num_fit_batches if args.num_fit_batches else len(train_loader)
        samples_per_batch = config.data.batch_size
        total_samples = n_batches * samples_per_batch

        print(f"\n[Time Estimate]")
        print(f"  Fitting {n_batches} batches × {samples_per_batch} samples = {total_samples} total")
        print(f"  Estimated time: {total_samples * 0.05:.0f}-{total_samples * 0.1:.0f} seconds")

        # Fit Laplace
        print("\n" + "="*60)
        print("Fitting Laplace (Computing Fisher)")
        print("="*60)
        fit_time = fit_laplace_models(
            laplace_x, laplace_adj,
            train_loader, sde_x, sde_adj,
            num_batches=args.num_fit_batches,
            device=device
        )
        print(f"\nTotal fitting time: {fit_time:.1f}s")

        # Save if requested
        if args.save_laplace:
            save_path = f"./checkpoints/{args.dataset}/laplace"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            laplace_x.save(save_path + '_x.pt')
            laplace_adj.save(save_path + '_adj.pt')

    # Print posterior stats
    print("\n" + "="*60)
    print("Posterior Statistics")
    print("="*60)
    post_std_x = laplace_x.get_posterior_std()
    post_std_adj = laplace_adj.get_posterior_std()
    print(f"ScoreNetworkX posterior std: min={post_std_x.min():.6f}, max={post_std_x.max():.6f}, mean={post_std_x.mean():.6f}")
    print(f"ScoreNetworkA posterior std: min={post_std_adj.min():.6f}, max={post_std_adj.max():.6f}, mean={post_std_adj.mean():.6f}")

    # Sample with uncertainty
    print("\n" + "="*60)
    print("Sampling with Uncertainty")
    print("="*60)

    # Load training graphs for init_flags
    train_loader, test_loader = load_data(config, get_graph_list=True)

    shape_x = (args.n_samples, config.data.max_node_num, config.data.max_feat_num)
    shape_adj = (args.n_samples, config.data.max_node_num, config.data.max_node_num)

    init_flags_tensor = init_flags(train_loader, config).to(device)[:args.n_samples]

    # Estimate sampling time
    print(f"\n[Time Estimate]")
    print(f"  {args.n_steps} steps × {args.n_uncertainty_samples} posterior samples")
    print(f"  ~{args.n_steps * args.n_uncertainty_samples * 2} forward passes")
    print(f"  Estimated time: {args.n_steps * 0.5:.0f}-{args.n_steps * 2:.0f} seconds")

    start_time = time.time()
    x_gen, adj_gen, uncertainties = uncertainty_aware_sampling(
        model_x, model_adj,
        laplace_x, laplace_adj,
        sde_x, sde_adj,
        init_flags_tensor,
        shape_x, shape_adj,
        n_steps=args.n_steps,
        n_uncertainty_samples=args.n_uncertainty_samples,
        device=device,
    )
    sample_time = time.time() - start_time

    x_pg = torch.stack(uncertainties["x_per_graph"], dim=0)   # [T,B]
    a_pg = torch.stack(uncertainties["adj_per_graph"], dim=0) # [T,B]

    print(f"\nSampling time: {sample_time:.1f}s")

    

    # Results
    print("\n" + "="*60)
    print("Results - Method 1")
    print("="*60)

    print("Global unc_x start/end:", uncertainties["x"][0], uncertainties["x"][-1])
    print("Per-graph x unc start:", x_pg[0].tolist())
    print("Per-graph x unc end:  ", x_pg[-1].tolist())
    print("Per-graph adj unc end:", a_pg[-1].tolist())

    T = x_pg.shape[0]
    start = int(0.7 * T)
    u_check = x_pg[start:].mean(0) + a_pg[start:].mean(0)
    print("u_check:", u_check.tolist())

    print(f"Generated x shape: {x_gen.shape}")
    print(f"Generated adj shape: {adj_gen.shape}")
    print(f"X value range: [{x_gen.min():.3f}, {x_gen.max():.3f}]")
    print(f"Adj value range: [{adj_gen.min():.3f}, {adj_gen.max():.3f}]")

    # Uncertainty trajectory
    unc_x = uncertainties['x']
    unc_adj = uncertainties['adj']
    print(f"\nUncertainty evolution:")
    print(f"  X: start={unc_x[0]:.4f}, middle={unc_x[len(unc_x)//2]:.4f}, end={unc_x[-1]:.4f}")
    print(f"  Adj: start={unc_adj[0]:.4f}, middle={unc_adj[len(unc_adj)//2]:.4f}, end={unc_adj[-1]:.4f}")

    # Convert to graphs
    adj_int = quantize(adj_gen.cpu())
    graphs = adjs_to_graphs(adj_int, True)
    print(f"\nGenerated {len(graphs)} valid graphs")

    for i, g in enumerate(graphs):
        print(f"  Graph {i}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # ===== Penalize invalid (NaN/Inf) samples =====
    invalid = ~torch.isfinite(
        x_gen.view(x_gen.shape[0], -1)
    ).all(dim=1)

    # Set uncertainty to +inf for invalid graphs
    u_graph = summarize_per_graph_uncertainty(uncertainties, late_fraction=0.3)
    u_graph[invalid] = float("inf")
    # =============================================
    print("Per-graph uncertainty (late steps):", u_graph.tolist())

    print("\n" + "="*60)
    print("Complete!")
    print("="*60)

    

    u_sem, vecs = graph_semantic_generative_uncertainty(
        model_x, model_adj,
        laplace_x, laplace_adj,
        sde_x, sde_adj,
        init_flags_tensor,
        shape_x, shape_adj,
        n_steps=args.n_steps,
        n_posterior=args.n_uncertainty_samples,
        seed=0,
        device=device,
        pick_index=0,
    )

    # Results
    print("\n" + "="*60)
    print("Results - Method 2")
    print("="*60)

    print("\nGraph-semantic generative uncertainty:", u_sem)
    print("Semantic embedding shape:", vecs.shape)

    print("\n" + "="*60)
    print("Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
