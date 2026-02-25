"""
Cross-step and cross-correction comparison of uncertainty filtering results.

Loads .npz files from a results directory and creates summary plots comparing:
- FCD / validity across diffusion step counts
- Combined vs X-only vs Adj-only filtering
- correct=True vs correct=False

Usage:
    python compare_comprehensive.py --results_dir ./results_comprehensive
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


def load_all_results(results_dir):
    """Load all .npz result files and organize by (steps, correction)."""
    results = {}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.npz'):
            continue
        path = os.path.join(results_dir, fname)
        data = dict(np.load(path, allow_pickle=True))

        # Parse filename: results_ZINC250k_100steps_correct.npz
        parts = fname.replace('.npz', '').split('_')
        steps = None
        corr = None
        for p in parts:
            if p.endswith('steps'):
                steps = int(p.replace('steps', ''))
            if p in ('correct', 'nocorrect', 'nocorrector'):
                corr = p

        if steps is not None and corr is not None:
            results[(steps, corr)] = data
            print(f"  Loaded: {fname} ({steps} steps, {corr})")

    return results


def plot_fcd_vs_steps(results, save_path):
    """
    Plot FCD vs diffusion steps for different filtering methods and correction settings.
    Shows FCD at 50% filtering (keeping the best 50% of molecules).
    """
    corr_modes = [
        ('correct', 'Standard (Langevin + Valence)'),
        ('nocorrect', 'No Valence Correction'),
        ('nocorrector', 'No Corrector'),
    ]
    # Only include modes that have data
    active_modes = [(k, l) for k, l in corr_modes if any(c == k for _, c in results.keys())]
    n_modes = max(len(active_modes), 1)
    fig, axes = plt.subplots(1, n_modes, figsize=(7 * n_modes, 6), squeeze=False)
    axes = axes[0]

    frac_idx = None  # Will find the index for 50% filtering

    for ax, (corr_key, corr_label) in zip(axes, active_modes):
        step_data = {}
        for (steps, corr), data in sorted(results.items()):
            if corr != corr_key:
                continue
            step_data[steps] = data

        if not step_data:
            ax.text(0.5, 0.5, f'No data for {corr_label}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(corr_label)
            continue

        steps_list = sorted(step_data.keys())

        # Find 50% fraction index
        fracs = list(step_data[steps_list[0]].get('filter_fractions', []))
        frac_idx = None
        for i, f in enumerate(fracs):
            if abs(float(f) - 0.5) < 0.01:
                frac_idx = i
                break
        if frac_idx is None:
            frac_idx = len(fracs) // 2

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

        for method in ['random', 'combined', 'x_only', 'adj_only']:
            fcd_key = f'{method}_fcd' if method != 'random' else 'random_fcd'
            fcds = []
            for s in steps_list:
                d = step_data[s]
                if fcd_key in d:
                    vals = list(d[fcd_key])
                    if frac_idx < len(vals):
                        fcds.append(float(vals[frac_idx]))
                    else:
                        fcds.append(float('nan'))
                else:
                    fcds.append(float('nan'))

            if method == 'random':
                fcd_std_key = 'random_fcd_std'
                stds = []
                for s in steps_list:
                    d = step_data[s]
                    if fcd_std_key in d:
                        vals = list(d[fcd_std_key])
                        if frac_idx < len(vals):
                            stds.append(float(vals[frac_idx]))
                        else:
                            stds.append(0)
                    else:
                        stds.append(0)
                ax.errorbar(steps_list, fcds, yerr=stds,
                            label=labels[method], color=colors[method],
                            marker='o', linewidth=2, capsize=3, linestyle='--')
            else:
                ax.plot(steps_list, fcds,
                        label=labels[method], color=colors[method],
                        marker='s', linewidth=2)

        ax.set_xlabel('Diffusion Steps')
        ax.set_ylabel('FCD (lower is better)')
        ax.set_title(f'{corr_label} (keeping best 50%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.set_xscale('log')
        ax.set_xticks(steps_list)
        ax.set_xticklabels([str(s) for s in steps_list])

    plt.suptitle('FCD vs Diffusion Steps by Filtering Method', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"FCD vs steps plot saved to: {save_path}")


def plot_fcd_across_fractions(results, save_path):
    """
    Plot FCD across filtering fractions for each step count.
    One subplot per step count, comparing methods.
    """
    all_steps = sorted(set(s for s, _ in results.keys()))
    n_steps = len(all_steps)

    corr_modes = [
        ('correct', 'Standard'),
        ('nocorrect', 'No Valence Corr.'),
        ('nocorrector', 'No Corrector'),
    ]
    active_modes = [(k, l) for k, l in corr_modes if any(c == k for _, c in results.keys())]
    n_rows = len(active_modes)

    fig, axes = plt.subplots(max(n_rows, 1), max(n_steps, 1),
                              figsize=(5 * max(n_steps, 1), 5 * max(n_rows, 1)),
                              squeeze=False)

    colors = {
        'random': '#95A5A6',
        'combined': '#27AE60',
        'x_only': '#2980B9',
        'adj_only': '#E67E22',
    }
    labels = {
        'random': 'Random',
        'combined': 'Combined',
        'x_only': 'X-Only',
        'adj_only': 'Adj-Only',
    }

    for row, (corr_key, corr_label) in enumerate(active_modes):
        for col, steps in enumerate(all_steps):
            ax = axes[row][col]
            key = (steps, corr_key)
            if key not in results:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(f'{steps} steps\n{corr_label}')
                continue

            data = results[key]
            fracs = list(data.get('filter_fractions', []))
            n_total = len(data.get('smiles', []))
            ns = [int(f * n_total) for f in fracs] if n_total > 0 else fracs

            for method in ['random', 'combined', 'x_only', 'adj_only']:
                fcd_key = f'{method}_fcd' if method != 'random' else 'random_fcd'
                if fcd_key not in data:
                    continue
                fcds = list(data[fcd_key])

                if method == 'random':
                    fcd_std_key = 'random_fcd_std'
                    stds = list(data.get(fcd_std_key, [0] * len(fcds)))
                    ax.errorbar(ns, fcds, yerr=stds,
                                label=labels[method], color=colors[method],
                                marker='o', linewidth=1.5, capsize=2, linestyle='--',
                                markersize=4)
                else:
                    ax.plot(ns, fcds,
                            label=labels[method], color=colors[method],
                            marker='s', linewidth=1.5, markersize=4)

            ax.set_xlabel('Nr. molecules kept')
            ax.set_ylabel('FCD')
            ax.set_title(f'{steps} steps\n{corr_label}')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

    plt.suptitle('FCD Across Filtering Fractions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"FCD across fractions plot saved to: {save_path}")


def plot_validity_vs_steps(results, save_path):
    """Plot validity vs steps for different filtering methods."""
    corr_modes = [
        ('correct', 'Standard (Langevin + Valence)'),
        ('nocorrect', 'No Valence Correction'),
        ('nocorrector', 'No Corrector'),
    ]
    active_modes = [(k, l) for k, l in corr_modes if any(c == k for _, c in results.keys())]
    n_modes = max(len(active_modes), 1)
    fig, axes = plt.subplots(1, n_modes, figsize=(7 * n_modes, 6), squeeze=False)
    axes = axes[0]

    for ax, (corr_key, corr_label) in zip(axes, active_modes):
        step_data = {}
        for (steps, corr), data in sorted(results.items()):
            if corr != corr_key:
                continue
            step_data[steps] = data

        if not step_data:
            ax.text(0.5, 0.5, f'No data for {corr_label}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(corr_label)
            continue

        steps_list = sorted(step_data.keys())

        # Find 50% fraction index
        fracs = list(step_data[steps_list[0]].get('filter_fractions', []))
        frac_idx = None
        for i, f in enumerate(fracs):
            if abs(float(f) - 0.5) < 0.01:
                frac_idx = i
                break
        if frac_idx is None:
            frac_idx = len(fracs) // 2

        colors = {
            'random': '#95A5A6',
            'combined': '#27AE60',
            'x_only': '#2980B9',
            'adj_only': '#E67E22',
        }
        labels = {
            'random': 'Random',
            'combined': 'Combined',
            'x_only': 'X-Only',
            'adj_only': 'Adj-Only',
        }

        for method in ['random', 'combined', 'x_only', 'adj_only']:
            val_key = f'{method}_validity' if method != 'random' else 'random_validity'
            vals_list = []
            for s in steps_list:
                d = step_data[s]
                if val_key in d:
                    vals = list(d[val_key])
                    if frac_idx < len(vals):
                        vals_list.append(float(vals[frac_idx]))
                    else:
                        vals_list.append(float('nan'))
                else:
                    vals_list.append(float('nan'))

            style = '--' if method == 'random' else '-'
            ax.plot(steps_list, vals_list,
                    label=labels[method], color=colors[method],
                    marker='s', linewidth=2, linestyle=style)

        ax.set_xlabel('Diffusion Steps')
        ax.set_ylabel('Validity')
        ax.set_title(f'{corr_label} (keeping best 50%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
        ax.set_xscale('log')
        ax.set_xticks(steps_list)
        ax.set_xticklabels([str(s) for s in steps_list])

    plt.suptitle('Validity vs Diffusion Steps by Filtering Method', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Validity vs steps plot saved to: {save_path}")


def plot_correction_comparison(results, save_path):
    """
    Direct comparison of correct=True vs correct=False for combined uncertainty.
    Shows FCD improvement (ours - random) for each setting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    all_steps = sorted(set(s for s, _ in results.keys()))

    # Left: FCD at 50% filtering
    ax = axes[0]
    for corr_key, corr_label, color, marker in [
        ('correct', 'Standard', '#27AE60', 's'),
        ('nocorrect', 'No Valence Corr.', '#E67E22', 'D'),
        ('nocorrector', 'No Corrector', '#8E44AD', '^'),
    ]:
        steps_list = sorted(s for s, c in results.keys() if c == corr_key)
        if not steps_list:
            continue

        fracs = list(results[(steps_list[0], corr_key)].get('filter_fractions', []))
        frac_idx = None
        for i, f in enumerate(fracs):
            if abs(float(f) - 0.5) < 0.01:
                frac_idx = i
                break
        if frac_idx is None:
            frac_idx = len(fracs) // 2

        # FCD improvement = random_fcd - combined_fcd (positive = ours is better)
        improvements = []
        for s in steps_list:
            d = results[(s, corr_key)]
            r_fcd = list(d.get('random_fcd', []))
            c_fcd = list(d.get('combined_fcd', []))
            if frac_idx < len(r_fcd) and frac_idx < len(c_fcd):
                improvements.append(float(r_fcd[frac_idx]) - float(c_fcd[frac_idx]))
            else:
                improvements.append(float('nan'))

        ax.plot(steps_list, improvements,
                label=corr_label, color=color, marker=marker, linewidth=2)

    ax.axhline(0, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Diffusion Steps')
    ax.set_ylabel('FCD Improvement (positive = ours better)')
    ax.set_title('FCD Improvement at 50% Filtering')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    if all_steps:
        ax.set_xticks(all_steps)
        ax.set_xticklabels([str(s) for s in all_steps])

    # Right: Overall validity comparison
    ax = axes[1]
    for corr_key, corr_label, color, marker in [
        ('correct', 'Standard', '#27AE60', 's'),
        ('nocorrect', 'No Valence Corr.', '#E67E22', 'D'),
        ('nocorrector', 'No Corrector', '#8E44AD', '^'),
    ]:
        steps_list = sorted(s for s, c in results.keys() if c == corr_key)
        if not steps_list:
            continue

        overall_validity = []
        for s in steps_list:
            d = results[(s, corr_key)]
            smiles = list(d.get('smiles', []))
            from rdkit import Chem
            n_valid = sum(1 for smi in smiles
                          if smi is not None and str(smi) not in ('None', '')
                          and Chem.MolFromSmiles(str(smi)) is not None)
            overall_validity.append(n_valid / len(smiles) if smiles else 0)

        ax.plot(steps_list, overall_validity,
                label=corr_label, color=color, marker=marker, linewidth=2)

    ax.set_xlabel('Diffusion Steps')
    ax.set_ylabel('Overall Validity')
    ax.set_title('Overall Validity (before filtering)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.set_xscale('log')
    if all_steps:
        ax.set_xticks(all_steps)
        ax.set_xticklabels([str(s) for s in all_steps])

    plt.suptitle('Correction Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Correction comparison plot saved to: {save_path}")


def plot_uncertainty_summary(results, save_path):
    """Plot uncertainty statistics across step counts."""
    n_corr_modes = sum(1 for k in ['correct', 'nocorrect', 'nocorrector']
                       if any(c == k for _, c in results.keys()))
    n_rows = max(n_corr_modes, 1)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), squeeze=False)

    unc_types = [
        ('unc_combined', 'Combined'),
        ('unc_x', 'X-Network'),
        ('unc_adj', 'Adj-Network'),
    ]

    corr_modes = [
        ('correct', 'Standard'),
        ('nocorrect', 'No Valence Corr.'),
        ('nocorrector', 'No Corrector'),
    ]
    active_modes = [(k, l) for k, l in corr_modes if any(c == k for _, c in results.keys())]
    for row, (corr_key, corr_label) in enumerate(active_modes):
        for col, (unc_key, unc_label) in enumerate(unc_types):
            ax = axes[row][col]

            steps_list = sorted(s for s, c in results.keys() if c == corr_key)
            if not steps_list:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes)
                continue

            means = []
            stds = []
            medians = []
            for s in steps_list:
                d = results[(s, corr_key)]
                arr = np.array(d.get(unc_key, []))
                finite = arr[np.isfinite(arr)]
                if len(finite) > 0:
                    means.append(np.mean(finite))
                    stds.append(np.std(finite))
                    medians.append(np.median(finite))
                else:
                    means.append(float('nan'))
                    stds.append(float('nan'))
                    medians.append(float('nan'))

            means = np.array(means)
            stds = np.array(stds)

            ax.errorbar(steps_list, means, yerr=stds,
                        color='#2980B9', marker='o', linewidth=2, capsize=4,
                        label='Mean +/- Std')
            ax.plot(steps_list, medians, color='#E67E22', marker='D',
                    linewidth=2, linestyle='--', label='Median')

            ax.set_xlabel('Diffusion Steps')
            ax.set_ylabel('Uncertainty (entropy)')
            ax.set_title(f'{unc_label}\n{corr_label}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            if steps_list:
                ax.set_xticks(steps_list)
                ax.set_xticklabels([str(s) for s in steps_list])

    plt.suptitle('Uncertainty Statistics Across Diffusion Steps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Uncertainty summary plot saved to: {save_path}")


def print_summary_table(results):
    """Print a comprehensive summary table."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 100)

    all_steps = sorted(set(s for s, _ in results.keys()))

    for corr_key, corr_label in [('correct', 'STANDARD (LANGEVIN + VALENCE)'),
                                    ('nocorrect', 'NO VALENCE CORRECTION'),
                                    ('nocorrector', 'NO CORRECTOR')]:
        print(f"\n--- {corr_label} ---")
        print(f"{'Steps':>6} | {'Validity':>10} | {'FCD_all':>10} | "
              f"{'FCD_50%_R':>10} | {'FCD_50%_C':>10} | {'FCD_50%_X':>10} | {'FCD_50%_A':>10} | "
              f"{'inf_c':>6} | {'inf_x':>6} | {'inf_a':>6}")
        print("-" * 110)

        for steps in all_steps:
            key = (steps, corr_key)
            if key not in results:
                continue

            d = results[key]
            smiles = list(d.get('smiles', []))
            from rdkit import Chem
            n_valid = sum(1 for smi in smiles
                          if smi is not None and str(smi) not in ('None', '')
                          and Chem.MolFromSmiles(str(smi)) is not None)
            validity = n_valid / len(smiles) if smiles else 0

            # Find 50% and 100% fraction indices
            fracs = list(d.get('filter_fractions', []))
            idx_50 = None
            idx_100 = None
            for i, f in enumerate(fracs):
                if abs(float(f) - 0.5) < 0.01:
                    idx_50 = i
                if abs(float(f) - 1.0) < 0.01:
                    idx_100 = i

            def get_fcd(key_name, idx):
                arr = list(d.get(key_name, []))
                if idx is not None and idx < len(arr):
                    return float(arr[idx])
                return float('nan')

            fcd_all = get_fcd('random_fcd', idx_100)
            fcd_r = get_fcd('random_fcd', idx_50)
            fcd_c = get_fcd('combined_fcd', idx_50)
            fcd_x = get_fcd('x_only_fcd', idx_50)
            fcd_a = get_fcd('adj_only_fcd', idx_50)

            unc_c = np.array(d.get('unc_combined', []))
            unc_x = np.array(d.get('unc_x', []))
            unc_a = np.array(d.get('unc_adj', []))

            print(f"{steps:>6} | {validity:>9.1%} | {fcd_all:>10.2f} | "
                  f"{fcd_r:>10.2f} | {fcd_c:>10.2f} | {fcd_x:>10.2f} | {fcd_a:>10.2f} | "
                  f"{np.isinf(unc_c).sum():>6} | {np.isinf(unc_x).sum():>6} | {np.isinf(unc_a).sum():>6}")


def plot_nspdk_vs_steps(results, save_path):
    """Plot NSPDK MMD vs diffusion steps for different filtering methods."""
    corr_modes = [
        ('correct', 'Standard'),
        ('nocorrect', 'No Valence Corr.'),
        ('nocorrector', 'No Corrector'),
    ]
    active_modes = [(k, l) for k, l in corr_modes if any(c == k for _, c in results.keys())]
    n_modes = max(len(active_modes), 1)
    fig, axes = plt.subplots(1, n_modes, figsize=(7 * n_modes, 6), squeeze=False)
    axes = axes[0]

    for ax, (corr_key, corr_label) in zip(axes, active_modes):
        step_data = {}
        for (steps, corr), data in sorted(results.items()):
            if corr != corr_key:
                continue
            step_data[steps] = data

        if not step_data:
            ax.text(0.5, 0.5, f'No data for {corr_label}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(corr_label)
            continue

        steps_list = sorted(step_data.keys())
        fracs = list(step_data[steps_list[0]].get('filter_fractions', []))
        frac_idx = None
        for i, f in enumerate(fracs):
            if abs(float(f) - 0.5) < 0.01:
                frac_idx = i
                break
        if frac_idx is None:
            frac_idx = len(fracs) // 2

        colors = {'random': '#95A5A6', 'combined': '#27AE60',
                  'x_only': '#2980B9', 'adj_only': '#E67E22'}
        labels = {'random': 'Random', 'combined': 'Combined',
                  'x_only': 'X-Only', 'adj_only': 'Adj-Only'}

        for method in ['random', 'combined', 'x_only', 'adj_only']:
            nspdk_key = f'{method}_nspdk' if method != 'random' else 'random_nspdk'
            vals = []
            for s in steps_list:
                d = step_data[s]
                if nspdk_key in d:
                    arr = list(d[nspdk_key])
                    if frac_idx < len(arr):
                        vals.append(float(arr[frac_idx]))
                    else:
                        vals.append(float('nan'))
                else:
                    vals.append(float('nan'))

            style = '--' if method == 'random' else '-'
            ax.plot(steps_list, vals, label=labels[method], color=colors[method],
                    marker='s', linewidth=2, linestyle=style)

        ax.set_xlabel('Diffusion Steps')
        ax.set_ylabel('NSPDK MMD (lower is better)')
        ax.set_title(f'{corr_label} (keeping best 50%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.set_xscale('log')
        ax.set_xticks(steps_list)
        ax.set_xticklabels([str(s) for s in steps_list])

    plt.suptitle('NSPDK MMD vs Diffusion Steps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"NSPDK vs steps plot saved to: {save_path}")


def plot_qed_sa_vs_steps(results, save_path):
    """Plot mean QED and mean SA vs diffusion steps."""
    corr_modes = [
        ('correct', 'Standard'),
        ('nocorrect', 'No Valence Corr.'),
        ('nocorrector', 'No Corrector'),
    ]
    active_modes = [(k, l) for k, l in corr_modes if any(c == k for _, c in results.keys())]
    n_modes = max(len(active_modes), 1)
    fig, axes = plt.subplots(2, n_modes, figsize=(7 * n_modes, 10), squeeze=False)

    for col, (corr_key, corr_label) in enumerate(active_modes):
        step_data = {}
        for (steps, corr), data in sorted(results.items()):
            if corr != corr_key:
                continue
            step_data[steps] = data

        if not step_data:
            for row in range(2):
                axes[row][col].text(0.5, 0.5, 'No data', ha='center', va='center',
                                    transform=axes[row][col].transAxes)
            continue

        steps_list = sorted(step_data.keys())
        fracs = list(step_data[steps_list[0]].get('filter_fractions', []))
        frac_idx = None
        for i, f in enumerate(fracs):
            if abs(float(f) - 0.5) < 0.01:
                frac_idx = i
                break
        if frac_idx is None:
            frac_idx = len(fracs) // 2

        colors = {'random': '#95A5A6', 'combined': '#27AE60',
                  'x_only': '#2980B9', 'adj_only': '#E67E22'}
        labels = {'random': 'Random', 'combined': 'Combined',
                  'x_only': 'X-Only', 'adj_only': 'Adj-Only'}

        for row, (metric, metric_label, higher_better) in enumerate([
            ('mean_qed', 'Mean QED', True),
            ('mean_sa', 'Mean SA Score', False),
        ]):
            ax = axes[row][col]
            for method in ['random', 'combined', 'x_only', 'adj_only']:
                key = f'{method}_{metric}' if method != 'random' else f'random_{metric}'
                vals = []
                for s in steps_list:
                    d = step_data[s]
                    if key in d:
                        arr = list(d[key])
                        if frac_idx < len(arr):
                            vals.append(float(arr[frac_idx]))
                        else:
                            vals.append(float('nan'))
                    else:
                        vals.append(float('nan'))

                style = '--' if method == 'random' else '-'
                ax.plot(steps_list, vals, label=labels[method], color=colors[method],
                        marker='s', linewidth=2, linestyle=style)

            arrow = '\u2191' if higher_better else '\u2193'
            ax.set_xlabel('Diffusion Steps')
            ax.set_ylabel(metric_label)
            ax.set_title(f'{metric_label} ({arrow}) - {corr_label}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_xticks(steps_list)
            ax.set_xticklabels([str(s) for s in steps_list])

    plt.suptitle('QED & SA Score vs Diffusion Steps (keeping best 50%)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"QED/SA vs steps plot saved to: {save_path}")


def plot_fcd_delta_heatmap(results, save_path):
    """
    FCD improvement heatmap: step count x filter fraction x mode.
    Shows FCD_random - FCD_combined (positive = filtering helps).
    """
    corr_modes = [
        ('correct', 'Standard'),
        ('nocorrect', 'No Valence Corr.'),
        ('nocorrector', 'No Corrector'),
    ]
    active_modes = [(k, l) for k, l in corr_modes if any(c == k for _, c in results.keys())]
    n_modes = max(len(active_modes), 1)
    fig, axes = plt.subplots(1, n_modes, figsize=(8 * n_modes, 6), squeeze=False)
    axes = axes[0]

    for ax, (corr_key, corr_label) in zip(axes, active_modes):
        step_data = {}
        for (steps, corr), data in sorted(results.items()):
            if corr != corr_key:
                continue
            step_data[steps] = data

        if not step_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(corr_label)
            continue

        steps_list = sorted(step_data.keys())
        fracs = list(step_data[steps_list[0]].get('filter_fractions', []))
        fracs_float = [float(f) for f in fracs]

        # Build heatmap: rows=steps, cols=fractions
        heatmap = np.full((len(steps_list), len(fracs_float)), np.nan)
        for i, s in enumerate(steps_list):
            d = step_data[s]
            r_fcd = list(d.get('random_fcd', []))
            c_fcd = list(d.get('combined_fcd', []))
            for j in range(min(len(r_fcd), len(c_fcd), len(fracs_float))):
                heatmap[i, j] = float(r_fcd[j]) - float(c_fcd[j])

        vmax = max(abs(np.nanmin(heatmap)), abs(np.nanmax(heatmap)), 0.1)
        im = ax.imshow(heatmap, aspect='auto', cmap='RdYlGn',
                        vmin=-vmax, vmax=vmax, origin='lower')
        plt.colorbar(im, ax=ax, label='FCD improvement (positive = filtering helps)')

        ax.set_yticks(range(len(steps_list)))
        ax.set_yticklabels([str(s) for s in steps_list])
        ax.set_ylabel('Diffusion Steps')

        # Show every other fraction label to avoid clutter
        ax.set_xticks(range(len(fracs_float)))
        ax.set_xticklabels([f'{f:.0%}' if i % 2 == 0 else '' for i, f in enumerate(fracs_float)],
                           rotation=45, ha='right')
        ax.set_xlabel('Filter Fraction (kept)')
        ax.set_title(f'{corr_label}')

        # Annotate cells with values
        for i in range(len(steps_list)):
            for j in range(len(fracs_float)):
                if not np.isnan(heatmap[i, j]):
                    ax.text(j, i, f'{heatmap[i, j]:.1f}', ha='center', va='center',
                            fontsize=7, color='black' if abs(heatmap[i, j]) < vmax * 0.6 else 'white')

    plt.suptitle('FCD Improvement Heatmap (Random FCD - Ours FCD)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"FCD delta heatmap saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing .npz result files')
    args = parser.parse_args()

    print("=" * 70)
    print("Comprehensive Results Comparison")
    print("=" * 70)
    print(f"Results directory: {args.results_dir}")

    # Load all results
    results = load_all_results(args.results_dir)

    if not results:
        print("No result files found!")
        return

    print(f"\nLoaded {len(results)} result files")

    # Print summary table
    print_summary_table(results)

    # Create all comparison plots
    plots_dir = os.path.join(args.results_dir, 'comparison_plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_fcd_vs_steps(results, os.path.join(plots_dir, 'fcd_vs_steps.png'))
    plot_fcd_across_fractions(results, os.path.join(plots_dir, 'fcd_across_fractions.png'))
    plot_validity_vs_steps(results, os.path.join(plots_dir, 'validity_vs_steps.png'))
    plot_correction_comparison(results, os.path.join(plots_dir, 'correction_comparison.png'))
    plot_uncertainty_summary(results, os.path.join(plots_dir, 'uncertainty_summary.png'))
    plot_nspdk_vs_steps(results, os.path.join(plots_dir, 'nspdk_vs_steps.png'))
    plot_qed_sa_vs_steps(results, os.path.join(plots_dir, 'qed_sa_vs_steps.png'))
    plot_fcd_delta_heatmap(results, os.path.join(plots_dir, 'fcd_delta_heatmap.png'))

    print(f"\nAll comparison plots saved to: {plots_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
