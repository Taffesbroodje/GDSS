"""
Molecular Semantic Uncertainty Pipeline for GDSS

This script implements the Jazbec et al. (2025) "Generative Uncertainty in
Diffusion Models" framework for molecular generation using GDSS.

Key features:
- Uses ChemNet embeddings as semantic likelihood (512-dim)
- Computes generative uncertainty via posterior sampling
- Detects low-quality/invalid molecules via high uncertainty

RUNTIME ESTIMATES:
- Fisher fitting: ~2-5 minutes (depends on num_batches)
- Sampling with ChemNet uncertainty: ~5-10 minutes (depends on n_posterior)

Usage:
    python run_mol_semantic_uncertainty.py --dataset ZINC250k --n_posterior 20
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
from sde import VPSDE, VESDE
from utils.loader import load_data
from utils.graph_utils import init_flags
from laplace_gdss_full import (
    create_laplace_models,
    fit_laplace_models,
)
from chemnet_semantic_uncertainty import (
    mol_semantic_generative_uncertainty,
    mol_semantic_per_graph_uncertainty,
    analyze_uncertainty_quality,
)


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
    parser.add_argument('--dataset', type=str, default='ZINC250k',
                        choices=['QM9', 'ZINC250k'])
    parser.add_argument('--num_fit_batches', type=int, default=20,
                        help='Number of batches for Fisher fitting')
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of molecules to generate')
    parser.add_argument('--n_steps', type=int, default=100,
                        help='Number of diffusion steps')
    parser.add_argument('--n_posterior', type=int, default=20,
                        help='Number of posterior samples for uncertainty')
    parser.add_argument('--prior_precision', type=float, default=1.0,
                        help='Prior precision (regularization)')
    parser.add_argument('--uncertainty_type', type=str, default='trace',
                        choices=['trace', 'entropy'],
                        help='Uncertainty metric type')
    parser.add_argument('--sigma_squared', type=float, default=1e-3,
                        help='Observation noise for entropy computation')
    parser.add_argument('--save_laplace', action='store_true',
                        help='Save fitted Laplace state')
    parser.add_argument('--load_laplace', type=str, default=None,
                        help='Load pre-fitted Laplace state')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("=" * 70)
    print("Molecular Semantic Uncertainty Pipeline (Jazbec et al. + ChemNet)")
    print("=" * 70)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    # Note: MPS has issues with float64 in VESDE, so we use CPU for now
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load checkpoint
    ckpt_path = f"./checkpoints/{args.dataset}/gdss_{args.dataset}.pth"
    if not os.path.exists(ckpt_path):
        # Try lowercase
        ckpt_path = f"./checkpoints/{args.dataset.lower()}/gdss_{args.dataset.lower()}.pth"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Please download the pretrained model first.")
        return

    model_x, model_adj, config = load_models_from_checkpoint(ckpt_path, device)

    print(f"\nDataset: {config.data.data}")
    print(f"Max nodes: {config.data.max_node_num}")
    print(f"Max features: {config.data.max_feat_num}")

    # Create SDEs based on config
    # Note: config uses beta_min/beta_max for both VP and VE SDEs
    # For VE, these are actually sigma_min/sigma_max
    if hasattr(config, 'sde') and hasattr(config.sde, 'x'):
        if config.sde.x.type == 'VP':
            sde_x = VPSDE(beta_min=config.sde.x.beta_min,
                          beta_max=config.sde.x.beta_max,
                          N=config.sde.x.num_scales)
        else:
            # VE SDE uses beta_min/beta_max as sigma_min/sigma_max
            sde_x = VESDE(sigma_min=config.sde.x.beta_min,
                          sigma_max=config.sde.x.beta_max,
                          N=config.sde.x.num_scales)

        if config.sde.adj.type == 'VP':
            sde_adj = VPSDE(beta_min=config.sde.adj.beta_min,
                            beta_max=config.sde.adj.beta_max,
                            N=config.sde.adj.num_scales)
        else:
            # VE SDE uses beta_min/beta_max as sigma_min/sigma_max
            sde_adj = VESDE(sigma_min=config.sde.adj.beta_min,
                            sigma_max=config.sde.adj.beta_max,
                            N=config.sde.adj.num_scales)
    else:
        # Default SDEs
        sde_x = VPSDE(beta_min=0.1, beta_max=1.0, N=1000)
        sde_adj = VPSDE(beta_min=0.1, beta_max=1.0, N=1000)

    # Create Laplace wrappers
    print("\n" + "=" * 70)
    print("Creating Laplace Wrappers")
    print("=" * 70)
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
        print("\n" + "=" * 70)
        print("Loading Training Data")
        print("=" * 70)
        train_loader, _ = load_data(config)
        print(f"Training batches: {len(train_loader)}")
        print(f"Batch size: {config.data.batch_size}")

        # Fit Laplace
        print("\n" + "=" * 70)
        print("Fitting Laplace (Computing Fisher)")
        print("=" * 70)
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
            print(f"Saved Laplace state to {save_path}")

    # Print posterior stats
    print("\n" + "=" * 70)
    print("Posterior Statistics")
    print("=" * 70)
    post_std_x = laplace_x.get_posterior_std()
    post_std_adj = laplace_adj.get_posterior_std()
    print(f"ScoreNetworkX posterior std: min={post_std_x.min():.6f}, "
          f"max={post_std_x.max():.6f}, mean={post_std_x.mean():.6f}")
    print(f"ScoreNetworkA posterior std: min={post_std_adj.min():.6f}, "
          f"max={post_std_adj.max():.6f}, mean={post_std_adj.mean():.6f}")

    # Load training graphs for init_flags
    train_loader, test_loader = load_data(config, get_graph_list=True)

    shape_x = (args.n_samples, config.data.max_node_num, config.data.max_feat_num)
    shape_adj = (args.n_samples, config.data.max_node_num, config.data.max_node_num)

    init_flags_tensor = init_flags(train_loader, config).to(device)[:args.n_samples]

    # Compute molecular semantic uncertainty
    print("\n" + "=" * 70)
    print("Computing Molecular Semantic Uncertainty (ChemNet)")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  - Posterior samples: {args.n_posterior}")
    print(f"  - Diffusion steps: {args.n_steps}")
    print(f"  - Uncertainty type: {args.uncertainty_type}")
    print(f"  - Molecules per sample: {args.n_samples}")

    start_time = time.time()

    # Per-molecule uncertainty
    uncertainties, embeddings, smiles_list = mol_semantic_per_graph_uncertainty(
        model_x, model_adj,
        laplace_x, laplace_adj,
        sde_x, sde_adj,
        init_flags_tensor,
        shape_x, shape_adj,
        dataset=args.dataset,
        n_steps=args.n_steps,
        n_posterior=args.n_posterior,
        seed=args.seed,
        device=device,
        uncertainty_type=args.uncertainty_type,
        sigma_squared=args.sigma_squared,
    )

    total_time = time.time() - start_time

    # Results
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    print(f"\nComputation time: {total_time:.1f}s")
    print(f"ChemNet embedding shape: {embeddings.shape}")

    print(f"\nPer-molecule semantic uncertainty:")
    for i, (u, smi_samples) in enumerate(zip(uncertainties, zip(*smiles_list))):
        valid_smiles = [s for s in smi_samples if s is not None]
        validity = len(valid_smiles) / len(smi_samples)
        representative = valid_smiles[0] if valid_smiles else "INVALID"
        if len(representative) > 50:
            representative = representative[:47] + "..."
        print(f"  Mol {i}: U={u:.4f}, validity={validity:.1%}, SMILES={representative}")

    # Analyze uncertainty quality
    analysis = analyze_uncertainty_quality(uncertainties, smiles_list)

    print(f"\nUncertainty Analysis:")
    print(f"  Mean uncertainty: {analysis['mean_uncertainty']:.4f}")
    print(f"  Std uncertainty: {analysis['std_uncertainty']:.4f}")
    print(f"  Min uncertainty: {analysis['min_uncertainty']:.4f}")
    print(f"  Max uncertainty: {analysis['max_uncertainty']:.4f}")
    print(f"  Mean validity rate: {analysis['mean_validity_rate']:.1%}")
    print(f"  Uncertainty-validity correlation: {analysis['uncertainty_validity_correlation']:.3f}")
    print(f"  Molecules with infinite uncertainty: {analysis['n_infinite_uncertainty']}")

    # Rank molecules by uncertainty (low uncertainty = high confidence)
    print("\n" + "=" * 70)
    print("Molecules Ranked by Confidence (Low Uncertainty)")
    print("=" * 70)

    sorted_indices = np.argsort(uncertainties)
    for rank, idx in enumerate(sorted_indices[:5]):
        u = uncertainties[idx]
        valid_smiles = [smiles_list[m][idx] for m in range(len(smiles_list))
                        if smiles_list[m][idx] is not None]
        smi = valid_smiles[0] if valid_smiles else "N/A"
        if len(smi) > 60:
            smi = smi[:57] + "..."
        print(f"  #{rank+1}: Mol {idx}, U={u:.4f}, SMILES={smi}")

    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    print("\nKey insight: Lower uncertainty indicates higher model confidence.")
    print("High uncertainty may indicate invalid or out-of-distribution molecules.")


if __name__ == "__main__":
    main()
