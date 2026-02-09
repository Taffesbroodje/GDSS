"""
Evaluation of Uncertainty-Based Filtering for Molecular Generation

This script reproduces Figure 3 from Jazbec et al. (2025) for molecular generation:
- Generate N molecules
- Compute uncertainty for each
- Filter by keeping n samples with lowest uncertainty
- Compare molecular metrics vs random baseline

Metrics:
- Validity: % of chemically valid molecules
- Uniqueness: % of unique molecules among valid ones
- Novelty: % of molecules not in training set
- FCD: Fréchet ChemNet Distance (lower is better)

Usage:
    python evaluate_uncertainty_filtering.py --dataset ZINC250k --n_molecules 500
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import defaultdict

sys.path.insert(0, '.')

from models.ScoreNetwork_X import ScoreNetworkX
from models.ScoreNetwork_A import ScoreNetworkA
from sde import VPSDE, VESDE
from utils.loader import load_data
from utils.graph_utils import init_flags
from utils.mol_utils import gen_mol, mols_to_smiles, load_smiles, canonicalize_smiles
from laplace_gdss_full import (
    create_laplace_models,
    fit_laplace_models,
    uncertainty_aware_sampling,
)
from chemnet_semantic_uncertainty import (
    ChemNetSemanticEncoder,
    mol_semantic_vector,
    semantic_uncertainty_trace,
)


def load_models_from_checkpoint(ckpt_path, device):
    """Load GDSS models from checkpoint."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    params_x = ckpt['params_x'].copy()
    params_adj = ckpt['params_adj'].copy()
    params_x.pop('model_type', None)
    params_adj.pop('model_type', None)

    model_x = ScoreNetworkX(**params_x).to(device)
    model_adj = ScoreNetworkA(**params_adj).to(device)

    model_x.load_state_dict(ckpt['x_state_dict'])
    model_adj.load_state_dict(ckpt['adj_state_dict'])

    if 'ema_x' in ckpt:
        from utils.ema import ExponentialMovingAverage
        ema_x = ExponentialMovingAverage(model_x.parameters(), decay=0.9999)
        ema_x.load_state_dict(ckpt['ema_x'])
        ema_x.copy_to(model_x.parameters())
        ema_adj = ExponentialMovingAverage(model_adj.parameters(), decay=0.9999)
        ema_adj.load_state_dict(ckpt['ema_adj'])
        ema_adj.copy_to(model_adj.parameters())

    return model_x, model_adj, ckpt['model_config']


def compute_molecular_metrics(
    smiles_list: List[str],
    train_smiles: List[str],
    test_smiles: List[str],
) -> Dict[str, float]:
    """
    Compute molecular generation metrics.

    Returns dict with: validity, uniqueness, novelty, fcd
    """
    from rdkit import Chem

    # Filter valid SMILES
    valid_smiles = []
    for smi in smiles_list:
        if smi is not None and len(smi) > 0:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)

    n_total = len(smiles_list)
    n_valid = len(valid_smiles)

    validity = n_valid / n_total if n_total > 0 else 0.0

    # Uniqueness
    unique_smiles = list(set(valid_smiles))
    uniqueness = len(unique_smiles) / n_valid if n_valid > 0 else 0.0

    # Novelty (not in training set)
    train_set = set(train_smiles)
    novel_smiles = [s for s in unique_smiles if s not in train_set]
    novelty = len(novel_smiles) / len(unique_smiles) if len(unique_smiles) > 0 else 0.0

    # FCD (if enough valid samples)
    fcd = float('inf')
    if len(valid_smiles) >= 10:
        try:
            from fcd_torch import FCD
            fcd_calculator = FCD(device='cpu', n_jobs=1)
            # Compare against test set
            fcd = fcd_calculator(valid_smiles, test_smiles[:len(valid_smiles)*2])
        except Exception as e:
            print(f"FCD computation failed: {e}")
            fcd = float('inf')

    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'fcd': fcd,
        'n_valid': n_valid,
        'n_unique': len(unique_smiles),
        'n_novel': len(novel_smiles),
    }


@torch.no_grad()
def generate_molecules_with_uncertainty(
    model_x, model_adj,
    laplace_x, laplace_adj,
    sde_x, sde_adj,
    init_flags_all,
    config,
    n_molecules: int,
    n_steps: int,
    n_posterior: int,
    batch_size: int,
    device: str,
    dataset: str,
) -> Tuple[List[str], np.ndarray]:
    """
    Generate molecules and compute per-molecule uncertainty.

    Returns:
        smiles_list: List of generated SMILES
        uncertainties: Array of uncertainties for each molecule
    """
    # Use CUDA for ChemNet if available (pass string, not torch.device)
    chemnet_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = ChemNetSemanticEncoder(device=chemnet_device)

    all_smiles = []
    all_uncertainties = []

    n_batches = (n_molecules + batch_size - 1) // batch_size

    print(f"\nGenerating {n_molecules} molecules in {n_batches} batches...")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_molecules)
        current_batch_size = end_idx - start_idx

        print(f"  Batch {batch_idx + 1}/{n_batches} ({current_batch_size} molecules)")

        # Get init flags for this batch
        init_flags_batch = init_flags_all[:current_batch_size].to(device)

        shape_x = (current_batch_size, config.data.max_node_num, config.data.max_feat_num)
        shape_adj = (current_batch_size, config.data.max_node_num, config.data.max_node_num)

        # CRITICAL: Generate FIXED noise for this batch (Jazbec et al. Algorithm 1)
        # All posterior samples will use the SAME noise z
        torch.manual_seed(batch_idx * 1000 + 42)  # Different seed per batch, but fixed within batch
        fixed_noise_x = sde_x.prior_sampling(shape_x)
        fixed_noise_adj = sde_adj.prior_sampling_sym(shape_adj)

        # Save MAP parameters
        theta_x_map = laplace_x._get_param_vector().clone()
        theta_a_map = laplace_adj._get_param_vector().clone()

        # Generate with MAP parameters first using FIXED noise
        laplace_x.fitted = False
        laplace_adj.fitted = False

        x_gen, adj_gen, _ = uncertainty_aware_sampling(
            model_x, model_adj,
            laplace_x, laplace_adj,
            sde_x, sde_adj,
            init_flags_batch,
            shape_x, shape_adj,
            n_steps=n_steps,
            n_uncertainty_samples=0,
            device=device,
            fixed_noise_x=fixed_noise_x,
            fixed_noise_adj=fixed_noise_adj,
        )

        laplace_x.fitted = True
        laplace_adj.fitted = True

        # Get MAP embeddings
        map_embeddings, map_smiles = mol_semantic_vector(x_gen, adj_gen, dataset, encoder)

        # Compute uncertainty via posterior sampling with FIXED noise z
        posterior_embeddings = [map_embeddings]

        for p_idx in range(n_posterior):
            # Sample posterior parameters θm ~ q(θ|D)
            theta_x = laplace_x.sample_parameters(1)[0]
            theta_a = laplace_adj.sample_parameters(1)[0]

            laplace_x._set_param_vector(theta_x)
            laplace_adj._set_param_vector(theta_a)

            laplace_x.fitted = False
            laplace_adj.fitted = False

            # Generate with FIXED noise z (same for all posterior samples!)
            x_gen_p, adj_gen_p, _ = uncertainty_aware_sampling(
                model_x, model_adj,
                laplace_x, laplace_adj,
                sde_x, sde_adj,
                init_flags_batch,
                shape_x, shape_adj,
                n_steps=n_steps,
                n_uncertainty_samples=0,
                device=device,
                fixed_noise_x=fixed_noise_x,  # FIXED noise!
                fixed_noise_adj=fixed_noise_adj,  # FIXED noise!
            )

            laplace_x.fitted = True
            laplace_adj.fitted = True

            embeddings_p, _ = mol_semantic_vector(x_gen_p, adj_gen_p, dataset, encoder)
            posterior_embeddings.append(embeddings_p)

        # Restore MAP parameters
        laplace_x._set_param_vector(theta_x_map)
        laplace_adj._set_param_vector(theta_a_map)

        # Compute per-molecule uncertainty
        posterior_embeddings = np.stack(posterior_embeddings, axis=0)  # [M+1, B, 512]

        batch_uncertainties = []
        for mol_idx in range(current_batch_size):
            emb = posterior_embeddings[:, mol_idx, :]  # [M+1, 512]
            u = semantic_uncertainty_trace(emb)
            batch_uncertainties.append(u)

        all_smiles.extend(map_smiles)
        all_uncertainties.extend(batch_uncertainties)

    return all_smiles, np.array(all_uncertainties)


def evaluate_filtering(
    smiles_list: List[str],
    uncertainties: np.ndarray,
    train_smiles: List[str],
    test_smiles: List[str],
    filter_fractions: List[float],
    n_runs: int = 5,
) -> Dict[str, Dict[str, List]]:
    """
    Evaluate metrics at different filtering levels.

    Returns dict with metrics for 'random' and 'uncertainty' methods.
    """
    n_total = len(smiles_list)

    results = {
        'random': defaultdict(list),
        'uncertainty': defaultdict(list),
    }

    filter_ns = [int(f * n_total) for f in filter_fractions]

    for n_keep in filter_ns:
        print(f"\n  Evaluating n={n_keep} (out of {n_total})")

        # Uncertainty-based filtering (deterministic)
        sorted_indices = np.argsort(uncertainties)
        unc_indices = sorted_indices[:n_keep]
        unc_smiles = [smiles_list[i] for i in unc_indices]
        unc_metrics = compute_molecular_metrics(unc_smiles, train_smiles, test_smiles)

        results['uncertainty']['n'].append(n_keep)
        for key, val in unc_metrics.items():
            results['uncertainty'][key].append(val)

        # Random baseline (multiple runs)
        random_metrics = defaultdict(list)
        for run in range(n_runs):
            np.random.seed(run + 42)
            random_indices = np.random.choice(n_total, n_keep, replace=False)
            random_smiles = [smiles_list[i] for i in random_indices]
            run_metrics = compute_molecular_metrics(random_smiles, train_smiles, test_smiles)
            for key, val in run_metrics.items():
                random_metrics[key].append(val)

        results['random']['n'].append(n_keep)
        for key in random_metrics:
            mean_val = np.mean(random_metrics[key])
            std_val = np.std(random_metrics[key])
            results['random'][key].append(mean_val)
            results['random'][f'{key}_std'].append(std_val)

    return results


def plot_filtering_results(
    results: Dict,
    save_path: str,
    n_total: int,
):
    """
    Create Figure 3-style plots for molecular generation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ('validity', 'Validity (\u2191)', True),
        ('uniqueness', 'Uniqueness (\u2191)', True),
        ('novelty', 'Novelty (\u2191)', True),
        ('fcd', 'FCD (\u2193)', False),
    ]

    colors = {
        'random': '#E74C3C',
        'uncertainty': '#27AE60',
    }

    for ax, (metric, label, higher_better) in zip(axes.flatten(), metrics):
        ns = results['random']['n']

        # Random baseline with error bars
        random_vals = results['random'][metric]
        random_std = results['random'].get(f'{metric}_std', [0] * len(random_vals))

        ax.errorbar(ns, random_vals, yerr=random_std,
                    label='Random', color=colors['random'],
                    marker='o', linewidth=2, capsize=3)

        # Uncertainty-based
        unc_vals = results['uncertainty'][metric]
        ax.plot(ns, unc_vals,
                label='Ours (ChemNet Unc.)', color=colors['uncertainty'],
                marker='s', linewidth=2)

        ax.set_xlabel(f'Nr. of molecules kept after filtering (out of {n_total})')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format y-axis
        if metric == 'fcd':
            ax.set_ylim(bottom=0)
        elif metric in ['validity', 'uniqueness', 'novelty']:
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    plt.suptitle('Uncertainty-Based Filtering for Molecular Generation\n(GDSS + ChemNet Semantic Uncertainty)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlot saved to: {save_path}")


def plot_uncertainty_quality_correlation(
    smiles_list: List[str],
    uncertainties: np.ndarray,
    save_path: str,
):
    """
    Plot showing relationship between uncertainty and molecule validity.
    """
    from rdkit import Chem

    # Compute validity for each molecule
    validities = []
    for smi in smiles_list:
        if smi is None or len(smi) == 0:
            validities.append(0)
        else:
            mol = Chem.MolFromSmiles(smi)
            validities.append(1 if mol is not None else 0)

    validities = np.array(validities)
    finite_mask = np.isfinite(uncertainties)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Histogram of uncertainties for valid vs invalid
    ax = axes[0]
    valid_unc = uncertainties[finite_mask & (validities == 1)]
    invalid_unc = uncertainties[finite_mask & (validities == 0)]

    if len(valid_unc) > 0:
        ax.hist(valid_unc, bins=30, alpha=0.6, label=f'Valid (n={len(valid_unc)})',
                color='#27AE60', density=True)
    if len(invalid_unc) > 0:
        ax.hist(invalid_unc, bins=30, alpha=0.6, label=f'Invalid (n={len(invalid_unc)})',
                color='#E74C3C', density=True)

    ax.set_xlabel('Semantic Uncertainty')
    ax.set_ylabel('Density')
    ax.set_title('Uncertainty Distribution by Validity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Validity rate vs uncertainty percentile
    ax = axes[1]
    n_bins = 10
    sorted_indices = np.argsort(uncertainties[finite_mask])
    sorted_validities = validities[finite_mask][sorted_indices]

    bin_size = len(sorted_validities) // n_bins
    percentiles = []
    validity_rates = []

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(sorted_validities)
        bin_validities = sorted_validities[start:end]
        validity_rates.append(np.mean(bin_validities))
        percentiles.append((i + 0.5) * (100 / n_bins))

    ax.bar(percentiles, validity_rates, width=8, color='#3498DB', alpha=0.7)
    ax.axhline(y=np.mean(validities), color='red', linestyle='--',
               label=f'Overall validity: {np.mean(validities):.1%}')
    ax.set_xlabel('Uncertainty Percentile (lower = more confident)')
    ax.set_ylabel('Validity Rate')
    ax.set_title('Validity vs Uncertainty Percentile')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Correlation Between Uncertainty and Molecule Quality',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Correlation plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ZINC250k',
                        choices=['QM9', 'ZINC250k'])
    parser.add_argument('--n_molecules', type=int, default=500,
                        help='Total number of molecules to generate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for generation')
    parser.add_argument('--n_steps', type=int, default=100,
                        help='Number of diffusion steps')
    parser.add_argument('--n_posterior', type=int, default=10,
                        help='Number of posterior samples for uncertainty')
    parser.add_argument('--num_fit_batches', type=int, default=10,
                        help='Number of batches for Fisher fitting')
    parser.add_argument('--prior_precision', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--load_laplace', type=str, default=None,
                        help='Load pre-fitted Laplace state')
    parser.add_argument('--save_laplace', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("Uncertainty-Based Filtering Evaluation for Molecular Generation")
    print("=" * 70)

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load models
    ckpt_path = f"./checkpoints/{args.dataset}/gdss_{args.dataset.lower()}.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"./checkpoints/{args.dataset}/gdss_{args.dataset}.pth"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    model_x, model_adj, config = load_models_from_checkpoint(ckpt_path, device)

    # Create SDEs
    if hasattr(config, 'sde') and hasattr(config.sde, 'x'):
        if config.sde.x.type == 'VP':
            sde_x = VPSDE(beta_min=config.sde.x.beta_min,
                          beta_max=config.sde.x.beta_max,
                          N=config.sde.x.num_scales)
        else:
            sde_x = VESDE(sigma_min=config.sde.x.beta_min,
                          sigma_max=config.sde.x.beta_max,
                          N=config.sde.x.num_scales)

        if config.sde.adj.type == 'VP':
            sde_adj = VPSDE(beta_min=config.sde.adj.beta_min,
                            beta_max=config.sde.adj.beta_max,
                            N=config.sde.adj.num_scales)
        else:
            sde_adj = VESDE(sigma_min=config.sde.adj.beta_min,
                            sigma_max=config.sde.adj.beta_max,
                            N=config.sde.adj.num_scales)
    else:
        sde_x = VPSDE(beta_min=0.1, beta_max=1.0, N=1000)
        sde_adj = VPSDE(beta_min=0.1, beta_max=1.0, N=1000)

    # Create Laplace wrappers
    laplace_x, laplace_adj = create_laplace_models(
        model_x, model_adj, prior_precision=args.prior_precision
    )

    # Load or fit Laplace
    if args.load_laplace and os.path.exists(args.load_laplace + '_x.pt'):
        print(f"\nLoading pre-fitted Laplace from {args.load_laplace}")
        laplace_x.load(args.load_laplace + '_x.pt')
        laplace_adj.load(args.load_laplace + '_adj.pt')
    else:
        train_loader, _ = load_data(config)
        fit_laplace_models(
            laplace_x, laplace_adj,
            train_loader, sde_x, sde_adj,
            num_batches=args.num_fit_batches,
            device=device
        )

        if args.save_laplace:
            save_path = f"./checkpoints/{args.dataset}/laplace"
            laplace_x.save(save_path + '_x.pt')
            laplace_adj.save(save_path + '_adj.pt')

    # Load reference SMILES
    print("\nLoading reference SMILES...")
    train_smiles, test_smiles = load_smiles(args.dataset)
    train_smiles = canonicalize_smiles(train_smiles)
    test_smiles = canonicalize_smiles(test_smiles)
    print(f"  Train: {len(train_smiles)}, Test: {len(test_smiles)}")

    # Load init flags
    train_loader, _ = load_data(config, get_graph_list=True)
    init_flags_all = init_flags(train_loader, config, args.n_molecules)

    # Generate molecules with uncertainty
    print("\n" + "=" * 70)
    print("Generating Molecules with Uncertainty Estimation")
    print("=" * 70)

    start_time = time.time()
    smiles_list, uncertainties = generate_molecules_with_uncertainty(
        model_x, model_adj,
        laplace_x, laplace_adj,
        sde_x, sde_adj,
        init_flags_all,
        config,
        n_molecules=args.n_molecules,
        n_steps=args.n_steps,
        n_posterior=args.n_posterior,
        batch_size=args.batch_size,
        device=device,
        dataset=args.dataset,
    )
    gen_time = time.time() - start_time
    print(f"\nGeneration time: {gen_time:.1f}s")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Generation Summary")
    print("=" * 70)
    valid_smiles = [s for s in smiles_list if s is not None]
    print(f"Total molecules: {len(smiles_list)}")
    print(f"Valid molecules: {len(valid_smiles)} ({len(valid_smiles)/len(smiles_list):.1%})")
    print(f"Uncertainty range: [{np.nanmin(uncertainties):.2f}, {np.nanmax(uncertainties):.2f}]")
    print(f"Uncertainty mean: {np.nanmean(uncertainties):.2f}")

    # Evaluate filtering at different levels
    print("\n" + "=" * 70)
    print("Evaluating Uncertainty-Based Filtering")
    print("=" * 70)

    filter_fractions = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = evaluate_filtering(
        smiles_list, uncertainties,
        train_smiles, test_smiles,
        filter_fractions,
        n_runs=5,
    )

    # Print results table
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"\n{'n':>6} | {'Method':>12} | {'Validity':>10} | {'Unique':>10} | {'Novelty':>10} | {'FCD':>10}")
    print("-" * 70)

    for i, n in enumerate(results['random']['n']):
        # Random
        print(f"{n:>6} | {'Random':>12} | "
              f"{results['random']['validity'][i]:>9.1%} | "
              f"{results['random']['uniqueness'][i]:>9.1%} | "
              f"{results['random']['novelty'][i]:>9.1%} | "
              f"{results['random']['fcd'][i]:>10.2f}")

        # Uncertainty
        print(f"{n:>6} | {'Ours':>12} | "
              f"{results['uncertainty']['validity'][i]:>9.1%} | "
              f"{results['uncertainty']['uniqueness'][i]:>9.1%} | "
              f"{results['uncertainty']['novelty'][i]:>9.1%} | "
              f"{results['uncertainty']['fcd'][i]:>10.2f}")
        print("-" * 70)

    # Create plots
    print("\n" + "=" * 70)
    print("Creating Visualization Plots")
    print("=" * 70)

    plot_filtering_results(
        results,
        os.path.join(args.output_dir, f'filtering_results_{args.dataset}.png'),
        args.n_molecules,
    )

    plot_uncertainty_quality_correlation(
        smiles_list, uncertainties,
        os.path.join(args.output_dir, f'uncertainty_correlation_{args.dataset}.png'),
    )

    # Save raw results
    np.savez(
        os.path.join(args.output_dir, f'results_{args.dataset}.npz'),
        smiles=smiles_list,
        uncertainties=uncertainties,
        filter_fractions=filter_fractions,
        random_validity=results['random']['validity'],
        random_uniqueness=results['random']['uniqueness'],
        random_novelty=results['random']['novelty'],
        random_fcd=results['random']['fcd'],
        unc_validity=results['uncertainty']['validity'],
        unc_uniqueness=results['uncertainty']['uniqueness'],
        unc_novelty=results['uncertainty']['novelty'],
        unc_fcd=results['uncertainty']['fcd'],
    )

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
