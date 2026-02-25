"""
Compare uncertainty filtering results across different diffusion step counts.

After running evaluate_uncertainty_steps.py at multiple step counts, this script
creates comparison plots showing how uncertainty filtering improves as generation
quality degrades (fewer steps).

Usage:
    python compare_steps.py --results_dir ./results_steps --dataset ZINC250k
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, '.')


def load_step_results(results_dir, dataset, step_counts):
    """Load results from multiple step counts."""
    all_results = {}
    for steps in step_counts:
        path = os.path.join(results_dir, f'results_{dataset}_{steps}steps.npz')
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            all_results[steps] = data
            print(f"  Loaded {steps} steps: {len(data['smiles'])} molecules")
        else:
            print(f"  Missing: {path}")
    return all_results


def plot_validity_vs_steps(all_results, save_path):
    """Show how validity degrades with fewer steps, and how uncertainty helps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    steps_list = sorted(all_results.keys())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(steps_list)))

    # 1. Overall validity vs steps
    ax = axes[0]
    overall_validity = []
    for steps in steps_list:
        data = all_results[steps]
        from rdkit import Chem
        smiles = data['smiles']
        n_valid = sum(1 for s in smiles if s is not None and str(s) not in ('None', '')
                      and Chem.MolFromSmiles(str(s)) is not None)
        overall_validity.append(n_valid / len(smiles))

    ax.plot(steps_list, overall_validity, 'ko-', linewidth=2, markersize=8)
    ax.set_xlabel('Diffusion Steps')
    ax.set_ylabel('Validity')
    ax.set_title('Overall Validity vs Steps')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # 2. Validity at 50% filtering vs steps (ours vs random)
    ax = axes[1]
    unc_validity_50 = []
    rand_validity_50 = []
    for steps in steps_list:
        data = all_results[steps]
        # Find the 50% filtering entry (or closest)
        fracs = data.get('filter_fractions', [0.5])
        idx_50 = np.argmin(np.abs(np.array(fracs) - 0.5))
        unc_validity_50.append(data['unc_validity'][idx_50])
        rand_validity_50.append(data['random_validity'][idx_50])

    ax.plot(steps_list, unc_validity_50, 's-', color='#27AE60', linewidth=2,
            markersize=8, label='Uncertainty (50%)')
    ax.plot(steps_list, rand_validity_50, 'o--', color='#E74C3C', linewidth=2,
            markersize=8, label='Random (50%)')
    ax.set_xlabel('Diffusion Steps')
    ax.set_ylabel('Validity (at 50% filtering)')
    ax.set_title('Filtering Validity vs Steps')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # 3. Validity improvement (ours - random) vs steps
    ax = axes[2]
    improvements = []
    for steps in steps_list:
        data = all_results[steps]
        # Average improvement across all filtering fractions
        unc_v = np.array(data['unc_validity'], dtype=float)
        rand_v = np.array(data['random_validity'], dtype=float)
        # Exclude 100% (no filtering)
        if len(unc_v) > 1:
            improvements.append(np.mean(unc_v[:-1] - rand_v[:-1]))
        else:
            improvements.append(0)

    ax.bar(range(len(steps_list)), improvements, color='#3498DB', alpha=0.7)
    ax.set_xticks(range(len(steps_list)))
    ax.set_xticklabels([str(s) for s in steps_list])
    ax.set_xlabel('Diffusion Steps')
    ax.set_ylabel('Validity Improvement\n(Ours - Random)')
    ax.set_title('Uncertainty Filtering Benefit')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Effect of Diffusion Steps on Uncertainty Filtering',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Steps comparison saved to: {save_path}")


def plot_filtering_across_steps(all_results, save_path):
    """Plot filtering curves for all step counts on the same axes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    steps_list = sorted(all_results.keys())
    cmap = plt.cm.coolwarm
    colors = {steps: cmap(i / max(len(steps_list) - 1, 1))
              for i, steps in enumerate(steps_list)}

    metrics = [
        ('validity', 'Validity', True),
        ('uniqueness', 'Uniqueness', True),
        ('novelty', 'Novelty', True),
        ('fcd', 'FCD', False),
    ]

    for ax, (metric, label, higher_better) in zip(axes.flatten(), metrics):
        for steps in steps_list:
            data = all_results[steps]
            n_total = len(data['smiles'])
            fracs = data.get('filter_fractions', np.linspace(0.5, 1.0, 6))
            ns = [int(f * n_total) for f in fracs]

            unc_vals = data[f'unc_{metric}']
            rand_vals = data[f'random_{metric}']

            ax.plot(np.array(ns) / n_total * 100, unc_vals, 's-',
                    color=colors[steps], linewidth=2, markersize=4,
                    label=f'{steps} steps (Ours)')
            ax.plot(np.array(ns) / n_total * 100, rand_vals, 'o--',
                    color=colors[steps], linewidth=1, markersize=3, alpha=0.5)

        ax.set_xlabel('% molecules kept')
        ax.set_ylabel(label)
        arrow = '\u2191' if higher_better else '\u2193'
        ax.set_title(f'{label} ({arrow})')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        if metric in ['validity', 'uniqueness', 'novelty']:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    plt.suptitle('Filtering Results Across Different Step Counts\n(solid=Uncertainty, dashed=Random)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Filtering comparison saved to: {save_path}")


def plot_x_vs_adj_across_steps(all_results, save_path):
    """Compare x-network vs adj-network uncertainty across step counts."""
    steps_list = sorted(all_results.keys())

    fig, axes = plt.subplots(1, len(steps_list), figsize=(5 * len(steps_list), 5))
    if len(steps_list) == 1:
        axes = [axes]

    for ax, steps in zip(axes, steps_list):
        data = all_results[steps]
        if 'unc_x' in data and 'unc_adj' in data:
            unc_x = data['unc_x']
            unc_adj = data['unc_adj']
            finite = np.isfinite(unc_x) & np.isfinite(unc_adj)

            if finite.sum() > 0:
                ax.hist2d(unc_x[finite], unc_adj[finite], bins=30, cmap='viridis')
                ax.set_xlabel('X-Network Uncertainty')
                ax.set_ylabel('Adj-Network Uncertainty')

                corr = np.corrcoef(unc_x[finite], unc_adj[finite])[0, 1]
                ax.set_title(f'{steps} steps\n(r={corr:.3f})')
            else:
                ax.set_title(f'{steps} steps\n(no valid data)')
        else:
            ax.text(0.5, 0.5, 'No separate\nuncertainties', ha='center',
                    va='center', transform=ax.transAxes)
            ax.set_title(f'{steps} steps')
        ax.grid(True, alpha=0.3)

    plt.suptitle('X vs Adj Network Uncertainty at Different Step Counts',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"X vs Adj comparison saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results_steps')
    parser.add_argument('--dataset', type=str, default='ZINC250k')
    parser.add_argument('--steps', type=str, default='50,100,200,500,1000',
                        help='Comma-separated list of step counts to compare')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'comparison')
    os.makedirs(args.output_dir, exist_ok=True)

    step_counts = [int(s) for s in args.steps.split(',')]

    print("=" * 70)
    print("Comparing Results Across Diffusion Steps")
    print("=" * 70)

    all_results = load_step_results(args.results_dir, args.dataset, step_counts)

    if len(all_results) < 2:
        print(f"\nNeed at least 2 step counts to compare. Found: {list(all_results.keys())}")
        if len(all_results) == 1:
            print("Run more experiments first.")
        return

    print(f"\nComparing: {sorted(all_results.keys())} steps")

    plot_validity_vs_steps(
        all_results,
        os.path.join(args.output_dir, f'validity_vs_steps_{args.dataset}.png'),
    )

    plot_filtering_across_steps(
        all_results,
        os.path.join(args.output_dir, f'filtering_across_steps_{args.dataset}.png'),
    )

    plot_x_vs_adj_across_steps(
        all_results,
        os.path.join(args.output_dir, f'x_vs_adj_across_steps_{args.dataset}.png'),
    )

    print(f"\nAll comparison plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
