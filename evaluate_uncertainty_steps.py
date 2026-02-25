"""
Uncertainty Evaluation with Variable Diffusion Steps

This script extends evaluate_uncertainty_filtering.py to:
1. Support variable diffusion steps (--num_scales) to degrade quality
2. Compute SEPARATE x-network and adj-network uncertainties
3. Fix the init_flags bug (fresh flags per batch)
4. Create 2x2 grid: high/low uncertainty from x-network vs adj-network

Usage:
    python evaluate_uncertainty_steps.py --dataset ZINC250k \
        --n_molecules 1000 --num_scales 100 --output_dir ./results_steps100
"""

import os
import sys
import time
import argparse
import pickle
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import defaultdict

sys.path.insert(0, '.')

from models.ScoreNetwork_X import ScoreNetworkX
from models.ScoreNetwork_A import ScoreNetworkA
from sde import VPSDE, VESDE
from utils.loader import load_data
from utils.graph_utils import init_flags, node_flags, graphs_to_tensor
from utils.mol_utils import gen_mol, mols_to_smiles, mols_to_nx, load_smiles, canonicalize_smiles
from evaluation.stats import nspdk_stats
from laplace_gdss_full import create_laplace_models, fit_laplace_models
from solver import get_pc_sampler
from chemnet_semantic_uncertainty import (
    ChemNetSemanticEncoder,
    mol_semantic_vector,
    semantic_uncertainty_entropy,
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


def compute_molecular_metrics(smiles_list, train_smiles, test_smiles, test_graph_list=None):
    """Compute molecular generation metrics including NSPDK MMD, mean QED, and mean SA."""
    from rdkit import Chem
    from rdkit.Chem import QED as QED_module

    valid_smiles = []
    valid_mols = []
    for smi in smiles_list:
        if smi is not None and len(smi) > 0:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
                valid_mols.append(mol)

    n_total = len(smiles_list)
    n_valid = len(valid_smiles)
    validity = n_valid / n_total if n_total > 0 else 0.0

    unique_smiles = list(set(valid_smiles))
    uniqueness = len(unique_smiles) / n_valid if n_valid > 0 else 0.0

    train_set = set(train_smiles)
    novel_smiles = [s for s in unique_smiles if s not in train_set]
    novelty = len(novel_smiles) / len(unique_smiles) if len(unique_smiles) > 0 else 0.0

    # FCD: always use full test set as reference for consistency across filter fractions
    fcd = float('inf')
    if len(valid_smiles) >= 10:
        try:
            from fcd_torch import FCD
            fcd_calculator = FCD(device='cpu', n_jobs=1)
            fcd = fcd_calculator(valid_smiles, test_smiles)
        except Exception as e:
            print(f"FCD computation failed: {e}")

    # NSPDK MMD: kernel-based metric, robust to small sample sizes
    nspdk = float('inf')
    if test_graph_list is not None and len(valid_mols) >= 10:
        try:
            gen_graphs = mols_to_nx(valid_mols)
            nspdk = nspdk_stats(test_graph_list, gen_graphs)
        except Exception as e:
            print(f"NSPDK computation failed: {e}")

    # Per-molecule property metrics
    mean_qed = float('nan')
    mean_sa = float('nan')
    if len(valid_mols) > 0:
        qed_vals = [QED_module.qed(mol) for mol in valid_mols]
        mean_qed = float(np.mean(qed_vals))

        try:
            from rdkit.Chem import RDConfig
            sa_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
            if sa_path not in sys.path:
                sys.path.append(sa_path)
            import sascorer
            sa_vals = [sascorer.calculateScore(mol) for mol in valid_mols]
            mean_sa = float(np.mean(sa_vals))
        except Exception as e:
            print(f"SA Score computation failed: {e}")

    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'fcd': fcd,
        'nspdk': nspdk,
        'mean_qed': mean_qed,
        'mean_sa': mean_sa,
        'n_valid': n_valid,
        'n_unique': len(unique_smiles),
        'n_novel': len(novel_smiles),
    }


def sample_init_flags_batch(graph_list, config, batch_size):
    """
    Sample fresh init_flags for a batch (fixes the init_flags bug).
    Each call samples random graph sizes from the training set.
    """
    max_node_num = config.data.max_node_num
    graph_tensor = graphs_to_tensor(graph_list, max_node_num)
    idx = np.random.randint(0, len(graph_list), batch_size)
    flags = node_flags(graph_tensor[idx])
    return flags


@torch.no_grad()
def generate_with_separate_uncertainties(
    model_x, model_adj,
    laplace_x, laplace_adj,
    sde_x, sde_adj,
    graph_list, config,
    n_molecules, n_posterior, batch_size,
    device, dataset, correct=True,
    disable_corrector=False,
):
    """
    Generate molecules with SEPARATE x-network and adj-network uncertainties.

    For each batch:
    1. Generate with MAP parameters → MAP embeddings
    2. Sample only x-network posterior (adj fixed) → x-uncertainty
    3. Sample only adj-network posterior (x fixed) → adj-uncertainty
    4. Sample both posteriors → combined uncertainty

    Returns:
        smiles_list, combined_uncertainties, x_uncertainties, adj_uncertainties
    """
    if torch.cuda.is_available():
        chemnet_device = 'cuda'
    elif torch.backends.mps.is_available():
        chemnet_device = 'mps'
    else:
        chemnet_device = 'cpu'
    encoder = ChemNetSemanticEncoder(device=chemnet_device)

    max_node_num = config.data.max_node_num
    shape_x = (batch_size, max_node_num, config.data.max_feat_num)
    shape_adj = (batch_size, max_node_num, max_node_num)

    device_str = str(device) if isinstance(device, torch.device) else device
    corrector_name = 'None' if disable_corrector else 'Langevin'
    sampling_fn = get_pc_sampler(
        sde_x=sde_x, sde_adj=sde_adj,
        shape_x=shape_x, shape_adj=shape_adj,
        predictor='Reverse', corrector=corrector_name,
        snr=0.2, scale_eps=0.9, n_steps=1,
        continuous=True, denoise=True, eps=1e-4,
        device=device_str,
    )

    all_smiles = []
    all_unc_combined = []
    all_unc_x = []
    all_unc_adj = []

    n_batches = (n_molecules + batch_size - 1) // batch_size

    print(f"\nGenerating {n_molecules} molecules in {n_batches} batches "
          f"(PC sampler, {sde_adj.N} diffusion steps)...")

    # Save MAP parameters
    theta_x_map = laplace_x._get_param_vector().clone()
    theta_a_map = laplace_adj._get_param_vector().clone()

    for batch_idx in range(n_batches):
        n_keep = min(batch_size, n_molecules - batch_idx * batch_size)

        print(f"\n  Batch {batch_idx + 1}/{n_batches} ({n_keep} molecules)")

        # FIX: Fresh init_flags per batch
        init_flags_batch = sample_init_flags_batch(graph_list, config, batch_size).to(device)

        # Fixed noise for this batch
        torch.manual_seed(batch_idx * 1000 + 42)
        fixed_noise_x = sde_x.prior_sampling(shape_x)
        fixed_noise_adj = sde_adj.prior_sampling_sym(shape_adj)

        # Restore MAP for base generation
        laplace_x._set_param_vector(theta_x_map)
        laplace_adj._set_param_vector(theta_a_map)

        # MAP generation
        print(f"    MAP sample...")
        x_gen, adj_gen, _ = sampling_fn(
            model_x, model_adj, init_flags_batch,
            fixed_noise_x=fixed_noise_x, fixed_noise_adj=fixed_noise_adj,
        )
        map_embeddings, map_smiles = mol_semantic_vector(
            x_gen[:n_keep], adj_gen[:n_keep], dataset, encoder, correct=correct
        )

        # --- Combined uncertainty (both networks vary) ---
        combined_embeddings = [map_embeddings]
        for p_idx in range(n_posterior):
            theta_x = laplace_x.sample_parameters(1)[0]
            theta_a = laplace_adj.sample_parameters(1)[0]
            laplace_x._set_param_vector(theta_x)
            laplace_adj._set_param_vector(theta_a)

            print(f"    Combined posterior {p_idx + 1}/{n_posterior}...")
            x_gen_p, adj_gen_p, _ = sampling_fn(
                model_x, model_adj, init_flags_batch,
                fixed_noise_x=fixed_noise_x, fixed_noise_adj=fixed_noise_adj,
            )
            emb_p, _ = mol_semantic_vector(
                x_gen_p[:n_keep], adj_gen_p[:n_keep], dataset, encoder, correct=correct
            )
            combined_embeddings.append(emb_p)

        laplace_x._set_param_vector(theta_x_map)
        laplace_adj._set_param_vector(theta_a_map)

        # --- X-only uncertainty (only x-network varies, adj fixed at MAP) ---
        x_only_embeddings = [map_embeddings]
        for p_idx in range(n_posterior):
            theta_x = laplace_x.sample_parameters(1)[0]
            laplace_x._set_param_vector(theta_x)
            # adj stays at MAP

            print(f"    X-only posterior {p_idx + 1}/{n_posterior}...")
            x_gen_p, adj_gen_p, _ = sampling_fn(
                model_x, model_adj, init_flags_batch,
                fixed_noise_x=fixed_noise_x, fixed_noise_adj=fixed_noise_adj,
            )
            emb_p, _ = mol_semantic_vector(
                x_gen_p[:n_keep], adj_gen_p[:n_keep], dataset, encoder, correct=correct
            )
            x_only_embeddings.append(emb_p)

        laplace_x._set_param_vector(theta_x_map)

        # --- Adj-only uncertainty (only adj-network varies, x fixed at MAP) ---
        adj_only_embeddings = [map_embeddings]
        for p_idx in range(n_posterior):
            theta_a = laplace_adj.sample_parameters(1)[0]
            laplace_adj._set_param_vector(theta_a)
            # x stays at MAP

            print(f"    Adj-only posterior {p_idx + 1}/{n_posterior}...")
            x_gen_p, adj_gen_p, _ = sampling_fn(
                model_x, model_adj, init_flags_batch,
                fixed_noise_x=fixed_noise_x, fixed_noise_adj=fixed_noise_adj,
            )
            emb_p, _ = mol_semantic_vector(
                x_gen_p[:n_keep], adj_gen_p[:n_keep], dataset, encoder, correct=correct
            )
            adj_only_embeddings.append(emb_p)

        laplace_adj._set_param_vector(theta_a_map)

        # Compute per-molecule uncertainties
        combined_embeddings = np.stack(combined_embeddings, axis=0)
        x_only_embeddings = np.stack(x_only_embeddings, axis=0)
        adj_only_embeddings = np.stack(adj_only_embeddings, axis=0)

        for mol_idx in range(n_keep):
            all_unc_combined.append(semantic_uncertainty_entropy(combined_embeddings[:, mol_idx, :]))
            all_unc_x.append(semantic_uncertainty_entropy(x_only_embeddings[:, mol_idx, :]))
            all_unc_adj.append(semantic_uncertainty_entropy(adj_only_embeddings[:, mol_idx, :]))

        all_smiles.extend(map_smiles)

    return (
        all_smiles,
        np.array(all_unc_combined),
        np.array(all_unc_x),
        np.array(all_unc_adj),
    )


def evaluate_filtering(smiles_list, uncertainties, train_smiles, test_smiles,
                       filter_fractions, test_graph_list=None, n_runs=5):
    """Evaluate metrics at different filtering levels."""
    n_total = len(smiles_list)
    results = {'random': defaultdict(list), 'uncertainty': defaultdict(list)}
    filter_ns = [int(f * n_total) for f in filter_fractions]

    for n_keep in filter_ns:
        print(f"\n  Evaluating n={n_keep} (out of {n_total})")

        sorted_indices = np.argsort(uncertainties)
        unc_indices = sorted_indices[:n_keep]
        unc_smiles = [smiles_list[i] for i in unc_indices]
        unc_metrics = compute_molecular_metrics(unc_smiles, train_smiles, test_smiles, test_graph_list)

        results['uncertainty']['n'].append(n_keep)
        for key, val in unc_metrics.items():
            results['uncertainty'][key].append(val)

        random_metrics = defaultdict(list)
        for run in range(n_runs):
            np.random.seed(run + 42)
            random_indices = np.random.choice(n_total, n_keep, replace=False)
            random_smiles = [smiles_list[i] for i in random_indices]
            run_metrics = compute_molecular_metrics(random_smiles, train_smiles, test_smiles, test_graph_list)
            for key, val in run_metrics.items():
                random_metrics[key].append(val)

        results['random']['n'].append(n_keep)
        for key in random_metrics:
            results['random'][key].append(np.mean(random_metrics[key]))
            results['random'][f'{key}_std'].append(np.std(random_metrics[key]))

    return results


def plot_2x2_grid(smiles_list, unc_x, unc_adj, save_path, dataset='ZINC250k'):
    """
    Create 2x2 grid comparing molecules based on x-network vs adj-network uncertainty.

    Quadrants:
    - Top-left:     Low X-unc, Low Adj-unc (confident both)
    - Top-right:    Low X-unc, High Adj-unc (x confident, adj uncertain)
    - Bottom-left:  High X-unc, Low Adj-unc (x uncertain, adj confident)
    - Bottom-right: High X-unc, High Adj-unc (uncertain both)
    """
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors, Draw

    finite_mask = np.isfinite(unc_x) & np.isfinite(unc_adj)
    valid_idx = np.where(finite_mask)[0]

    # Median split
    x_median = np.median(unc_x[finite_mask])
    adj_median = np.median(unc_adj[finite_mask])

    quadrants = {
        'Low X-unc, Low Adj-unc\n(Most Confident)': (unc_x < x_median) & (unc_adj < adj_median) & finite_mask,
        'Low X-unc, High Adj-unc\n(Adj Uncertain)': (unc_x < x_median) & (unc_adj >= adj_median) & finite_mask,
        'High X-unc, Low Adj-unc\n(X Uncertain)': (unc_x >= x_median) & (unc_adj < adj_median) & finite_mask,
        'High X-unc, High Adj-unc\n(Least Confident)': (unc_x >= x_median) & (unc_adj >= adj_median) & finite_mask,
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 18))
    n_show = 8

    for ax, (title, mask) in zip(axes.flatten(), quadrants.items()):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            ax.text(0.5, 0.5, 'No molecules', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
            continue

        # Pick representative molecules (middle of the group by combined uncertainty)
        combined = unc_x[idx] + unc_adj[idx]
        sorted_order = np.argsort(combined)
        # Take from middle for representative sample
        mid = len(sorted_order) // 2
        sample_idx = idx[sorted_order[max(0, mid - n_show//2) : mid + n_show//2]]

        # Compute metrics for this quadrant
        q_smiles = [str(smiles_list[i]) for i in idx if str(smiles_list[i]) not in ('None', '')]
        q_valid = [s for s in q_smiles if Chem.MolFromSmiles(s) is not None]
        validity = len(q_valid) / len(q_smiles) if len(q_smiles) > 0 else 0

        qed_vals = []
        for s in q_valid[:500]:  # limit for speed
            mol = Chem.MolFromSmiles(s)
            if mol:
                qed_vals.append(QED.qed(mol))
        mean_qed = np.mean(qed_vals) if qed_vals else 0

        # Draw molecules
        mols = []
        legends = []
        for i in sample_idx:
            smi = str(smiles_list[i])
            mol = Chem.MolFromSmiles(smi) if smi not in ('None', '') else None
            if mol is not None:
                mols.append(mol)
                legends.append(f'Ux={unc_x[i]:.0f} Ua={unc_adj[i]:.0f}')

        if len(mols) > 0:
            img = Draw.MolsToGridImage(mols[:n_show], molsPerRow=4, subImgSize=(300, 250),
                                       legends=legends[:n_show])
            ax.imshow(img)

        stats_text = f'n={len(idx)} | Valid={validity:.1%} | QED={mean_qed:.3f}'
        ax.set_title(f'{title}\n{stats_text}', fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle('2×2 Grid: X-Network vs Adj-Network Uncertainty',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"2x2 grid saved to: {save_path}")


def plot_scatter_x_vs_adj(unc_x, unc_adj, smiles_list, save_path):
    """Scatter plot of x-uncertainty vs adj-uncertainty, colored by validity."""
    from rdkit import Chem

    finite_mask = np.isfinite(unc_x) & np.isfinite(unc_adj)

    valid_flags = np.zeros(len(smiles_list))
    for i, smi in enumerate(smiles_list):
        if smi is not None and str(smi) not in ('None', ''):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                valid_flags[i] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter colored by validity
    ax = axes[0]
    valid_mask = (valid_flags == 1) & finite_mask
    invalid_mask = (valid_flags == 0) & finite_mask

    if invalid_mask.sum() > 0:
        ax.scatter(unc_x[invalid_mask], unc_adj[invalid_mask],
                   alpha=0.3, s=10, c='#E74C3C', label=f'Invalid ({invalid_mask.sum()})')
    if valid_mask.sum() > 0:
        ax.scatter(unc_x[valid_mask], unc_adj[valid_mask],
                   alpha=0.1, s=10, c='#27AE60', label=f'Valid ({valid_mask.sum()})')

    ax.set_xlabel('X-Network Uncertainty')
    ax.set_ylabel('Adj-Network Uncertainty')
    ax.set_title('X vs Adj Uncertainty (colored by validity)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: 2D histogram
    ax = axes[1]
    h = ax.hist2d(unc_x[finite_mask], unc_adj[finite_mask],
                  bins=50, cmap='viridis')
    plt.colorbar(h[3], ax=ax, label='Count')
    ax.set_xlabel('X-Network Uncertainty')
    ax.set_ylabel('Adj-Network Uncertainty')
    ax.set_title('Uncertainty Joint Distribution')
    ax.grid(True, alpha=0.3)

    # Add median lines
    x_med = np.median(unc_x[finite_mask])
    adj_med = np.median(unc_adj[finite_mask])
    for a in axes:
        a.axvline(x_med, color='white', linestyle='--', alpha=0.7, linewidth=1)
        a.axhline(adj_med, color='white', linestyle='--', alpha=0.7, linewidth=1)

    plt.suptitle('X-Network vs Adj-Network Uncertainty Comparison',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"X vs Adj scatter saved to: {save_path}")


def plot_filtering_results(results, save_path, n_total, num_scales):
    """Create filtering comparison plots."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))

    metrics = [
        ('validity', 'Validity', True),
        ('uniqueness', 'Uniqueness', True),
        ('novelty', 'Novelty', True),
        ('fcd', 'FCD', False),
        ('nspdk', 'NSPDK MMD', False),
        ('mean_qed', 'Mean QED', True),
    ]

    colors = {'random': '#E74C3C', 'uncertainty': '#27AE60'}

    for ax, (metric, label, higher_better) in zip(axes.flatten(), metrics):
        ns = results['random']['n']

        random_vals = results['random'][metric]
        random_std = results['random'].get(f'{metric}_std', [0] * len(random_vals))
        ax.errorbar(ns, random_vals, yerr=random_std,
                    label='Random', color=colors['random'],
                    marker='o', linewidth=2, capsize=3)

        unc_vals = results['uncertainty'][metric]
        ax.plot(ns, unc_vals,
                label='Ours (Uncertainty)', color=colors['uncertainty'],
                marker='s', linewidth=2)

        ax.set_xlabel(f'Nr. molecules kept (out of {n_total})')
        ax.set_ylabel(label)
        arrow = '\u2191' if higher_better else '\u2193'
        ax.set_title(f'{label} ({arrow})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if metric in ['validity', 'uniqueness', 'novelty']:
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        elif metric in ['fcd', 'nspdk']:
            ax.set_ylim(bottom=0)

    plt.suptitle(f'Uncertainty-Based Filtering ({num_scales} diffusion steps)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Filtering plot saved to: {save_path}")


def plot_multi_uncertainty_filtering(all_results, save_path, n_total, num_scales, correct):
    """
    Create comparison plots for Combined vs X-only vs Adj-only filtering.

    all_results: dict with keys 'combined', 'x_only', 'adj_only',
                 each containing the output of evaluate_filtering().
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    metrics = [
        ('validity', 'Validity', True),
        ('uniqueness', 'Uniqueness', True),
        ('novelty', 'Novelty', True),
        ('fcd', 'FCD', False),
        ('nspdk', 'NSPDK MMD', False),
        ('mean_qed', 'Mean QED', True),
    ]

    colors = {
        'random': '#95A5A6',
        'combined': '#27AE60',
        'x_only': '#2980B9',
        'adj_only': '#E67E22',
    }
    labels = {
        'random': 'Random',
        'combined': 'Combined (X+Adj)',
        'x_only': 'X-Network Only',
        'adj_only': 'Adj-Network Only',
    }

    for ax, (metric, label, higher_better) in zip(axes.flatten(), metrics):
        # Random baseline (same for all)
        r = all_results['combined']['random']
        ns = r['n']
        random_vals = r[metric]
        random_std = r.get(f'{metric}_std', [0] * len(random_vals))
        ax.errorbar(ns, random_vals, yerr=random_std,
                    label=labels['random'], color=colors['random'],
                    marker='o', linewidth=2, capsize=3, linestyle='--')

        # Each uncertainty type
        for unc_type in ['combined', 'x_only', 'adj_only']:
            vals = all_results[unc_type]['uncertainty'][metric]
            ax.plot(ns, vals,
                    label=labels[unc_type], color=colors[unc_type],
                    marker='s', linewidth=2)

        ax.set_xlabel(f'Nr. molecules kept (out of {n_total})')
        ax.set_ylabel(label)
        arrow = '\u2191' if higher_better else '\u2193'
        ax.set_title(f'{label} ({arrow})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        if metric in ['validity', 'uniqueness', 'novelty']:
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        elif metric in ['fcd', 'nspdk']:
            ax.set_ylim(bottom=0)

    corr_label = "with correction" if correct else "without correction"
    plt.suptitle(f'Filtering: Combined vs X-Only vs Adj-Only\n'
                 f'({num_scales} steps, {corr_label})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Multi-uncertainty filtering plot saved to: {save_path}")


def plot_uncertainty_distributions(unc_combined, unc_x, unc_adj, smiles_list,
                                   save_path, num_scales, correct):
    """Plot uncertainty value distributions and correlations between uncertainty types."""
    from rdkit import Chem

    valid_flags = np.array([
        1.0 if (s is not None and str(s) not in ('None', '') and
                Chem.MolFromSmiles(str(s)) is not None)
        else 0.0
        for s in smiles_list
    ])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: histograms of each uncertainty type
    for ax, (arr, name, color) in zip(axes[0], [
        (unc_combined, 'Combined', '#27AE60'),
        (unc_x, 'X-Network', '#2980B9'),
        (unc_adj, 'Adj-Network', '#E67E22'),
    ]):
        finite = arr[np.isfinite(arr)]
        valid_vals = arr[np.isfinite(arr) & (valid_flags == 1)]
        invalid_vals = arr[np.isfinite(arr) & (valid_flags == 0)]

        if len(valid_vals) > 0:
            ax.hist(valid_vals, bins=40, alpha=0.6, color=color, label=f'Valid ({len(valid_vals)})')
        if len(invalid_vals) > 0:
            ax.hist(invalid_vals, bins=40, alpha=0.6, color='#E74C3C', label=f'Invalid ({len(invalid_vals)})')

        n_inf = np.isinf(arr).sum()
        ax.set_title(f'{name} Uncertainty\n'
                     f'range=[{finite.min():.0f}, {finite.max():.0f}], '
                     f'inf={n_inf}')
        ax.set_xlabel('Uncertainty (entropy)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Bottom row: pairwise scatter plots
    pairs = [
        (unc_x, unc_adj, 'X-Network', 'Adj-Network'),
        (unc_x, unc_combined, 'X-Network', 'Combined'),
        (unc_adj, unc_combined, 'Adj-Network', 'Combined'),
    ]
    for ax, (arr1, arr2, name1, name2) in zip(axes[1], pairs):
        mask = np.isfinite(arr1) & np.isfinite(arr2)
        valid_mask = mask & (valid_flags == 1)
        invalid_mask = mask & (valid_flags == 0)

        if invalid_mask.sum() > 0:
            ax.scatter(arr1[invalid_mask], arr2[invalid_mask],
                       alpha=0.4, s=15, c='#E74C3C', label='Invalid', zorder=2)
        if valid_mask.sum() > 0:
            ax.scatter(arr1[valid_mask], arr2[valid_mask],
                       alpha=0.2, s=15, c='#27AE60', label='Valid', zorder=1)

        if mask.sum() > 2:
            corr = np.corrcoef(arr1[mask], arr2[mask])[0, 1]
            ax.set_title(f'{name1} vs {name2}\n(r={corr:.3f})')
        else:
            ax.set_title(f'{name1} vs {name2}')
        ax.set_xlabel(name1)
        ax.set_ylabel(name2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    corr_label = "with correction" if correct else "without correction"
    plt.suptitle(f'Uncertainty Distributions ({num_scales} steps, {corr_label})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Uncertainty distributions plot saved to: {save_path}")


def plot_precision_recall(smiles_list, uncertainties, train_smiles, save_path, num_scales):
    """Plot precision/recall for validity-based filtering."""
    from rdkit import Chem

    n_total = len(smiles_list)
    sorted_idx = np.argsort(uncertainties)

    valid_flags = np.zeros(n_total)
    for i, smi in enumerate(smiles_list):
        if smi is not None and str(smi) not in ('None', ''):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                valid_flags[i] = 1.0

    n_valid_total = valid_flags.sum()
    overall_validity = n_valid_total / n_total

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Precision-Recall curve
    ax = axes[0]
    fracs = np.linspace(0.05, 1.0, 50)
    precisions = []
    recalls = []
    for frac in fracs:
        n_keep = int(frac * n_total)
        kept = sorted_idx[:n_keep]
        tp = valid_flags[kept].sum()
        precisions.append(tp / n_keep if n_keep > 0 else 0)
        recalls.append(tp / n_valid_total if n_valid_total > 0 else 0)

    ax.plot(recalls, precisions, 'o-', color='#27AE60', markersize=2, linewidth=2,
            label='Uncertainty')
    ax.plot([0, 1], [overall_validity, overall_validity], '--', color='#E74C3C',
            linewidth=2, label=f'Random (P={overall_validity:.2%})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision (Validity)')
    ax.set_title(f'Precision-Recall ({num_scales} steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    # 2. Validity vs fraction kept
    ax = axes[1]
    validity_at_frac = []
    for frac in fracs:
        n_keep = int(frac * n_total)
        kept = sorted_idx[:n_keep]
        validity_at_frac.append(valid_flags[kept].sum() / n_keep if n_keep > 0 else 0)

    ax.plot(fracs * 100, validity_at_frac, 'o-', color='#27AE60', markersize=2, linewidth=2,
            label='Uncertainty')
    ax.axhline(overall_validity, color='#E74C3C', linestyle='--', linewidth=2,
               label=f'Overall: {overall_validity:.2%}')
    ax.set_xlabel('% of molecules kept')
    ax.set_ylabel('Validity')
    ax.set_title('Validity vs Filtering Fraction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 3. Cumulative invalid molecules removed
    ax = axes[2]
    invalid_flags = 1 - valid_flags
    n_invalid_total = invalid_flags.sum()
    invalid_removed = []
    for frac in fracs:
        n_keep = int(frac * n_total)
        kept = sorted_idx[:n_keep]
        n_invalid_kept = invalid_flags[kept].sum()
        frac_invalid_removed = 1 - (n_invalid_kept / n_invalid_total) if n_invalid_total > 0 else 1
        invalid_removed.append(frac_invalid_removed)

    ax.plot(fracs * 100, invalid_removed, 'o-', color='#27AE60', markersize=2, linewidth=2,
            label='Uncertainty')
    # Random baseline: keeping frac% removes (1-frac)% of invalids
    random_removed = [1 - f for f in fracs]
    ax.plot(fracs * 100, random_removed, '--', color='#E74C3C', linewidth=2, label='Random')
    ax.set_xlabel('% of molecules kept')
    ax.set_ylabel('% of invalid molecules removed')
    ax.set_title('Invalid Molecule Removal')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Precision & Recall Analysis ({num_scales} diffusion steps, '
                 f'overall validity={overall_validity:.1%})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Precision/recall saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ZINC250k')
    parser.add_argument('--n_molecules', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--n_posterior', type=int, default=10)
    parser.add_argument('--num_fit_batches', type=int, default=10)
    parser.add_argument('--num_scales', type=int, default=1000,
                        help='Number of diffusion steps (default 1000, reduce to degrade quality)')
    parser.add_argument('--prior_precision', type=float, default=1.0)
    parser.add_argument('--no_correction', action='store_true')
    parser.add_argument('--disable_corrector', action='store_true',
                        help='Disable Langevin corrector in diffusion loop (uses NoneCorrector)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./results_steps')
    parser.add_argument('--load_laplace', type=str, default=None)
    parser.add_argument('--save_laplace', action='store_true')
    parser.add_argument('--multilayer_laplace', action='store_true',
                        help='Use all 3 MLP layers for Laplace instead of just the last layer')
    parser.add_argument('--skip_nspdk', action='store_true',
                        help='Skip NSPDK MMD computation (very slow on large test sets)')
    args = parser.parse_args()

    print("=" * 70)
    print(f"Uncertainty Evaluation with {args.num_scales} Diffusion Steps")
    print(f"  Corrector: {'Disabled (NoneCorrector)' if args.disable_corrector else 'Langevin'}")
    print(f"  Valence correction: {'No' if args.no_correction else 'Yes'}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir = args.output_dir.rstrip('/')
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load models
    ckpt_path = f"./checkpoints/{args.dataset}/gdss_{args.dataset.lower()}.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"./checkpoints/{args.dataset}/gdss_{args.dataset}.pth"
    model_x, model_adj, config = load_models_from_checkpoint(ckpt_path, device)

    # Create SDEs with variable number of scales
    num_scales = args.num_scales
    print(f"\nUsing {num_scales} diffusion steps (default=1000)")

    if hasattr(config, 'sde') and hasattr(config.sde, 'x'):
        if config.sde.x.type == 'VP':
            sde_x = VPSDE(beta_min=config.sde.x.beta_min,
                          beta_max=config.sde.x.beta_max, N=num_scales)
        else:
            sde_x = VESDE(sigma_min=config.sde.x.beta_min,
                          sigma_max=config.sde.x.beta_max, N=num_scales)
        if config.sde.adj.type == 'VP':
            sde_adj = VPSDE(beta_min=config.sde.adj.beta_min,
                            beta_max=config.sde.adj.beta_max, N=num_scales)
        else:
            sde_adj = VESDE(sigma_min=config.sde.adj.beta_min,
                            sigma_max=config.sde.adj.beta_max, N=num_scales)
    else:
        sde_x = VPSDE(beta_min=0.1, beta_max=1.0, N=num_scales)
        sde_adj = VPSDE(beta_min=0.1, beta_max=1.0, N=num_scales)

    # Create Laplace wrappers
    if args.multilayer_laplace:
        last_layer_name = 'final.linears'  # All 3 MLP layers
        print(f"\n[Multilayer Laplace] Using all MLP layers (final.linears.0, .1, .2)")
    else:
        last_layer_name = 'final.linears.2'  # Last layer only
    laplace_x, laplace_adj = create_laplace_models(
        model_x, model_adj, prior_precision=args.prior_precision,
        last_layer_name=last_layer_name
    )

    # Load or fit Laplace (uses full 1000-step SDE for fitting regardless)
    if args.load_laplace and os.path.exists(args.load_laplace + '_x.pt'):
        print(f"\nLoading pre-fitted Laplace from {args.load_laplace}")
        laplace_x.load(args.load_laplace + '_x.pt')
        laplace_adj.load(args.load_laplace + '_adj.pt')
    else:
        # Fit with original SDE scales
        if hasattr(config, 'sde') and hasattr(config.sde, 'x'):
            if config.sde.x.type == 'VP':
                fit_sde_x = VPSDE(beta_min=config.sde.x.beta_min,
                                  beta_max=config.sde.x.beta_max,
                                  N=config.sde.x.num_scales)
            else:
                fit_sde_x = VESDE(sigma_min=config.sde.x.beta_min,
                                  sigma_max=config.sde.x.beta_max,
                                  N=config.sde.x.num_scales)
            if config.sde.adj.type == 'VP':
                fit_sde_adj = VPSDE(beta_min=config.sde.adj.beta_min,
                                    beta_max=config.sde.adj.beta_max,
                                    N=config.sde.adj.num_scales)
            else:
                fit_sde_adj = VESDE(sigma_min=config.sde.adj.beta_min,
                                    sigma_max=config.sde.adj.beta_max,
                                    N=config.sde.adj.num_scales)
        else:
            fit_sde_x = VPSDE(beta_min=0.1, beta_max=1.0, N=1000)
            fit_sde_adj = VPSDE(beta_min=0.1, beta_max=1.0, N=1000)

        train_loader, _ = load_data(config)
        fit_laplace_models(
            laplace_x, laplace_adj,
            train_loader, fit_sde_x, fit_sde_adj,
            num_batches=args.num_fit_batches, device=device
        )

        if args.save_laplace:
            suffix = "_multilayer" if args.multilayer_laplace else ""
            save_path = f"./checkpoints/{args.dataset}/laplace{suffix}"
            laplace_x.save(save_path + '_x.pt')
            laplace_adj.save(save_path + '_adj.pt')

    # Load reference SMILES
    print("\nLoading reference SMILES...")
    train_smiles, test_smiles = load_smiles(args.dataset)
    train_smiles = canonicalize_smiles(train_smiles)
    test_smiles = canonicalize_smiles(test_smiles)
    print(f"  Train: {len(train_smiles)}, Test: {len(test_smiles)}")

    # Load test graphs for NSPDK MMD
    test_graph_path = f'data/{args.dataset.lower()}_test_nx.pkl'
    test_graph_list = None
    if os.path.exists(test_graph_path):
        with open(test_graph_path, 'rb') as f:
            test_graph_list = pickle.load(f)
        print(f"  Test graphs for NSPDK: {len(test_graph_list)}")
    else:
        print(f"  Warning: {test_graph_path} not found, NSPDK will be skipped")

    # Load graph list for init_flags
    train_loader, graph_list = load_data(config, get_graph_list=True)

    # Generate with separate uncertainties
    print("\n" + "=" * 70)
    print(f"Generating {args.n_molecules} Molecules ({num_scales} steps)")
    print("=" * 70)

    start_time = time.time()
    smiles_list, unc_combined, unc_x, unc_adj = generate_with_separate_uncertainties(
        model_x, model_adj,
        laplace_x, laplace_adj,
        sde_x, sde_adj,
        graph_list, config,
        n_molecules=args.n_molecules,
        n_posterior=args.n_posterior,
        batch_size=args.batch_size,
        device=device,
        dataset=args.dataset,
        correct=not args.no_correction,
        disable_corrector=args.disable_corrector,
    )
    gen_time = time.time() - start_time
    print(f"\nGeneration time: {gen_time:.1f}s")

    # Summary
    valid_smiles = [s for s in smiles_list if s is not None and str(s) not in ('None', '')]
    from rdkit import Chem
    valid_mols = [s for s in valid_smiles if Chem.MolFromSmiles(s) is not None]

    print(f"\n" + "=" * 70)
    print(f"Generation Summary ({num_scales} steps)")
    print("=" * 70)
    print(f"Total molecules: {len(smiles_list)}")
    print(f"Valid molecules: {len(valid_mols)} ({len(valid_mols)/len(smiles_list):.1%})")
    print(f"Combined uncertainty: [{np.nanmin(unc_combined):.2f}, {np.nanmax(unc_combined):.2f}], "
          f"mean={np.nanmean(unc_combined):.2f}")
    print(f"X-network uncertainty: [{np.nanmin(unc_x):.2f}, {np.nanmax(unc_x):.2f}], "
          f"mean={np.nanmean(unc_x):.2f}")
    print(f"Adj-network uncertainty: [{np.nanmin(unc_adj):.2f}, {np.nanmax(unc_adj):.2f}], "
          f"mean={np.nanmean(unc_adj):.2f}")

    # Evaluate filtering for ALL uncertainty types
    print("\n" + "=" * 70)
    print("Evaluating Uncertainty-Based Filtering")
    print("=" * 70)

    correct = not args.no_correction
    filter_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]

    all_filter_results = {}
    for unc_name, unc_arr in [('combined', unc_combined),
                               ('x_only', unc_x),
                               ('adj_only', unc_adj)]:
        print(f"\n--- Filtering by {unc_name} uncertainty ---")
        all_filter_results[unc_name] = evaluate_filtering(
            smiles_list, unc_arr,
            train_smiles, test_smiles,
            filter_fractions, test_graph_list=None if args.skip_nspdk else test_graph_list, n_runs=5,
        )

    # Print results table for each uncertainty type
    for unc_name in ['combined', 'x_only', 'adj_only']:
        results = all_filter_results[unc_name]
        print(f"\n--- {unc_name.upper()} Uncertainty Filtering ---")
        print(f"{'n':>6} | {'Method':>12} | {'Validity':>10} | {'Unique':>10} | {'Novelty':>10} | "
              f"{'FCD':>10} | {'NSPDK':>10} | {'QED':>8} | {'SA':>8}")
        print("-" * 100)
        for i, n in enumerate(results['random']['n']):
            r = results['random']
            print(f"{n:>6} | {'Random':>12} | "
                  f"{r['validity'][i]:>9.1%} | "
                  f"{r['uniqueness'][i]:>9.1%} | "
                  f"{r['novelty'][i]:>9.1%} | "
                  f"{r['fcd'][i]:>10.2f} | "
                  f"{r['nspdk'][i]:>10.4f} | "
                  f"{r['mean_qed'][i]:>8.4f} | "
                  f"{r['mean_sa'][i]:>8.3f}")
            u = results['uncertainty']
            print(f"{n:>6} | {'Ours':>12} | "
                  f"{u['validity'][i]:>9.1%} | "
                  f"{u['uniqueness'][i]:>9.1%} | "
                  f"{u['novelty'][i]:>9.1%} | "
                  f"{u['fcd'][i]:>10.2f} | "
                  f"{u['nspdk'][i]:>10.4f} | "
                  f"{u['mean_qed'][i]:>8.4f} | "
                  f"{u['mean_sa'][i]:>8.3f}")
            print("-" * 100)

    # Create plots
    print("\n" + "=" * 70)
    print("Creating Visualization Plots")
    print("=" * 70)

    if args.disable_corrector:
        corr_tag = "nocorrector"
    elif correct:
        corr_tag = "correct"
    else:
        corr_tag = "nocorrect"

    if args.multilayer_laplace:
        corr_tag = corr_tag + "_multilayer"

    # 1. Combined-only filtering (backward compatible)
    plot_filtering_results(
        all_filter_results['combined'],
        os.path.join(args.output_dir, f'filtering_{args.dataset}_{num_scales}steps_{corr_tag}.png'),
        args.n_molecules, num_scales,
    )

    # 2. Multi-uncertainty comparison (Combined vs X vs Adj vs Random)
    plot_multi_uncertainty_filtering(
        all_filter_results,
        os.path.join(args.output_dir, f'filtering_comparison_{args.dataset}_{num_scales}steps_{corr_tag}.png'),
        args.n_molecules, num_scales, correct,
    )

    # 3. Uncertainty distributions and correlations
    plot_uncertainty_distributions(
        unc_combined, unc_x, unc_adj, smiles_list,
        os.path.join(args.output_dir, f'uncertainty_distributions_{args.dataset}_{num_scales}steps_{corr_tag}.png'),
        num_scales, correct,
    )

    # 4. Precision/recall
    plot_precision_recall(
        smiles_list, unc_combined, train_smiles,
        os.path.join(args.output_dir, f'precision_recall_{args.dataset}_{num_scales}steps_{corr_tag}.png'),
        num_scales,
    )

    # 5. 2x2 grid
    plot_2x2_grid(
        smiles_list, unc_x, unc_adj,
        os.path.join(args.output_dir, f'grid_2x2_{args.dataset}_{num_scales}steps_{corr_tag}.png'),
        args.dataset,
    )

    # 6. X vs Adj scatter
    plot_scatter_x_vs_adj(
        unc_x, unc_adj, smiles_list,
        os.path.join(args.output_dir, f'scatter_x_vs_adj_{args.dataset}_{num_scales}steps_{corr_tag}.png'),
    )

    # Save comprehensive raw results
    save_dict = dict(
        smiles=smiles_list,
        unc_combined=unc_combined,
        unc_x=unc_x,
        unc_adj=unc_adj,
        num_scales=num_scales,
        correct=correct,
        filter_fractions=filter_fractions,
    )
    # Save random baseline once (same for all uncertainty types since it's random)
    r_random = all_filter_results['combined']['random']
    save_dict['random_validity'] = r_random['validity']
    save_dict['random_validity_std'] = r_random.get('validity_std', [])
    save_dict['random_uniqueness'] = r_random['uniqueness']
    save_dict['random_uniqueness_std'] = r_random.get('uniqueness_std', [])
    save_dict['random_fcd'] = r_random['fcd']
    save_dict['random_fcd_std'] = r_random.get('fcd_std', [])
    save_dict['random_novelty'] = r_random['novelty']
    save_dict['random_novelty_std'] = r_random.get('novelty_std', [])
    save_dict['random_nspdk'] = r_random['nspdk']
    save_dict['random_nspdk_std'] = r_random.get('nspdk_std', [])
    save_dict['random_mean_qed'] = r_random['mean_qed']
    save_dict['random_mean_qed_std'] = r_random.get('mean_qed_std', [])
    save_dict['random_mean_sa'] = r_random['mean_sa']
    save_dict['random_mean_sa_std'] = r_random.get('mean_sa_std', [])

    # Save filtering results for each uncertainty type
    for unc_name in ['combined', 'x_only', 'adj_only']:
        r = all_filter_results[unc_name]
        save_dict[f'{unc_name}_validity'] = r['uncertainty']['validity']
        save_dict[f'{unc_name}_uniqueness'] = r['uncertainty']['uniqueness']
        save_dict[f'{unc_name}_novelty'] = r['uncertainty']['novelty']
        save_dict[f'{unc_name}_fcd'] = r['uncertainty']['fcd']
        save_dict[f'{unc_name}_nspdk'] = r['uncertainty']['nspdk']
        save_dict[f'{unc_name}_mean_qed'] = r['uncertainty']['mean_qed']
        save_dict[f'{unc_name}_mean_sa'] = r['uncertainty']['mean_sa']

    np.savez(
        os.path.join(args.output_dir, f'results_{args.dataset}_{num_scales}steps_{corr_tag}.npz'),
        **save_dict,
    )

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
