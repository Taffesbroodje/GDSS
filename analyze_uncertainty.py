"""
Comprehensive Uncertainty Analysis for Molecular Generation

Analyzes saved results from evaluate_uncertainty_filtering.py to produce:
1. Precision/recall curves for uncertainty-based filtering
2. Enhanced uncertainty-validity/quality correlation plots
3. Molecular property analysis (QED, SA, weight, logP) vs uncertainty
4. Per-decile quality breakdown
5. 2x2 grid: high/low uncertainty molecule examples

Usage:
    python analyze_uncertainty.py --results_dir ./results --dataset ZINC250k
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def load_results(results_dir, dataset):
    """Load saved results."""
    path = os.path.join(results_dir, f'results_{dataset}.npz')
    data = np.load(path, allow_pickle=True)
    smiles = data['smiles']
    uncertainties = data['uncertainties']
    return smiles, uncertainties, data


def compute_mol_properties(smiles_list):
    """Compute molecular properties for each SMILES."""
    properties = {
        'qed': [],
        'sa': [],
        'logp': [],
        'mw': [],
        'num_rings': [],
        'num_atoms': [],
        'valid': [],
    }

    for smi in smiles_list:
        if smi is None or str(smi) in ('None', ''):
            for k in properties:
                properties[k].append(np.nan)
            continue

        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            for k in properties:
                properties[k].append(np.nan)
            continue

        properties['valid'].append(1.0)
        properties['qed'].append(QED.qed(mol))

        try:
            from rdkit.Chem import RDConfig
            import sys
            sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
            import sascorer
            properties['sa'].append(sascorer.calculateScore(mol))
        except Exception:
            properties['sa'].append(np.nan)

        properties['logp'].append(Descriptors.MolLogP(mol))
        properties['mw'].append(Descriptors.MolWt(mol))
        properties['num_rings'].append(Descriptors.RingCount(mol))
        properties['num_atoms'].append(mol.GetNumHeavyAtoms())

    return {k: np.array(v) for k, v in properties.items()}


def precision_recall_curves(smiles, uncertainties, train_smiles, test_smiles, output_dir):
    """
    Plot precision and recall curves for uncertainty-based filtering.

    Defines "good" molecules by multiple criteria:
    - Valid (RDKit parseable)
    - Unique (not duplicated)
    - Novel (not in training set)
    - Drug-like (QED > 0.5)
    """
    from fcd_torch import FCD

    n_total = len(smiles)
    sorted_idx = np.argsort(uncertainties)  # lowest uncertainty first

    # Compute per-molecule quality flags
    valid_flags = np.zeros(n_total)
    unique_set = set()
    unique_flags = np.zeros(n_total)
    novel_flags = np.zeros(n_total)
    train_set = set(train_smiles) if train_smiles is not None else set()

    props = compute_mol_properties(smiles)
    druglike_flags = np.array([1.0 if q > 0.5 else 0.0 for q in props['qed']])

    for i, smi in enumerate(smiles):
        if smi is not None and str(smi) not in ('None', ''):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                valid_flags[i] = 1.0
                can_smi = Chem.MolToSmiles(mol)
                if can_smi not in unique_set:
                    unique_set.add(can_smi)
                    unique_flags[i] = 1.0
                if can_smi not in train_set:
                    novel_flags[i] = 1.0

    # Define quality criteria
    criteria = {
        'Valid': valid_flags,
        'Valid & Unique': valid_flags * unique_flags,
        'Valid & Novel': valid_flags * novel_flags,
        'Drug-like (QED>0.5)': druglike_flags,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for ax_idx, (name, good_flags) in enumerate(criteria.items()):
        ax = axes.flatten()[ax_idx]
        n_good_total = good_flags.sum()

        if n_good_total == 0:
            ax.text(0.5, 0.5, f'No "{name}" molecules found',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(name)
            continue

        # Compute precision/recall at different thresholds
        fractions = np.linspace(0.1, 1.0, 50)
        precisions_unc = []
        recalls_unc = []
        precisions_rand = []
        recalls_rand = []

        for frac in fractions:
            n_keep = int(frac * n_total)

            # Uncertainty-based
            kept_idx = sorted_idx[:n_keep]
            tp = good_flags[kept_idx].sum()
            precision = tp / n_keep if n_keep > 0 else 0
            recall = tp / n_good_total if n_good_total > 0 else 0
            precisions_unc.append(precision)
            recalls_unc.append(recall)

            # Random baseline (expectation)
            expected_tp = frac * n_good_total
            precision_rand = expected_tp / n_keep if n_keep > 0 else 0
            recall_rand = expected_tp / n_good_total if n_good_total > 0 else 0
            precisions_rand.append(precision_rand)
            recalls_rand.append(recall_rand)

        ax.plot(recalls_unc, precisions_unc, 'o-', color='#27AE60',
                label='Uncertainty', markersize=2, linewidth=2)
        ax.plot(recalls_rand, precisions_rand, '--', color='#E74C3C',
                label='Random', linewidth=2)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'{name}\n(total: {int(n_good_total)}/{n_total})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)

    plt.suptitle('Precision-Recall: Uncertainty vs Random Filtering',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'precision_recall.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Precision-recall plot saved to: {path}")


def uncertainty_property_correlation(smiles, uncertainties, output_dir):
    """
    Plot molecular properties vs uncertainty deciles.
    """
    props = compute_mol_properties(smiles)

    finite_mask = np.isfinite(uncertainties)
    n = finite_mask.sum()

    sorted_idx = np.argsort(uncertainties[finite_mask])
    n_deciles = 10
    decile_size = n // n_deciles

    prop_names = ['qed', 'sa', 'logp', 'mw', 'num_rings', 'num_atoms']
    prop_labels = ['QED (Drug-likeness)', 'SA Score (lower=easier)',
                   'LogP', 'Molecular Weight', 'Ring Count', 'Heavy Atom Count']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, pname, plabel in zip(axes.flatten(), prop_names, prop_labels):
        prop_vals = props[pname][finite_mask]
        valid_prop_mask = np.isfinite(prop_vals)

        decile_means = []
        decile_stds = []
        decile_labels = []

        for d in range(n_deciles):
            start = d * decile_size
            end = start + decile_size if d < n_deciles - 1 else n
            idx = sorted_idx[start:end]
            vals = prop_vals[idx]
            vals = vals[np.isfinite(vals)]

            if len(vals) > 0:
                decile_means.append(np.mean(vals))
                decile_stds.append(np.std(vals))
            else:
                decile_means.append(np.nan)
                decile_stds.append(0)
            decile_labels.append(f'{d*10}-{(d+1)*10}%')

        x = np.arange(n_deciles)
        ax.bar(x, decile_means, yerr=decile_stds, color='#3498DB',
               alpha=0.7, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(decile_labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Uncertainty Percentile (low → high)')
        ax.set_ylabel(plabel)
        ax.set_title(plabel)
        ax.grid(True, alpha=0.3, axis='y')

        # Add overall mean line
        all_valid = prop_vals[valid_prop_mask]
        if len(all_valid) > 0:
            ax.axhline(y=np.mean(all_valid), color='red', linestyle='--',
                       alpha=0.7, label=f'Mean: {np.mean(all_valid):.2f}')
            ax.legend(fontsize=8)

    plt.suptitle('Molecular Properties by Uncertainty Decile\n(Left = Most Confident, Right = Least Confident)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'property_vs_uncertainty.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Property correlation plot saved to: {path}")


def uncertainty_distribution_analysis(smiles, uncertainties, output_dir):
    """
    Detailed analysis of uncertainty distribution and its relationship to quality.
    """
    props = compute_mol_properties(smiles)
    finite_mask = np.isfinite(uncertainties)

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig)

    # 1. Uncertainty histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(uncertainties[finite_mask], bins=50, color='#3498DB', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Semantic Uncertainty (Entropy)')
    ax1.set_ylabel('Count')
    ax1.set_title('Uncertainty Distribution')
    ax1.axvline(np.median(uncertainties[finite_mask]), color='red', linestyle='--',
                label=f'Median: {np.median(uncertainties[finite_mask]):.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. QED vs Uncertainty scatter
    ax2 = fig.add_subplot(gs[0, 1])
    qed_valid = np.isfinite(props['qed']) & finite_mask
    if qed_valid.sum() > 0:
        ax2.scatter(uncertainties[qed_valid], props['qed'][qed_valid],
                    alpha=0.1, s=5, c='#2C3E50')
        # Add binned means
        n_bins = 20
        sorted_idx = np.argsort(uncertainties[qed_valid])
        bin_size = len(sorted_idx) // n_bins
        bin_centers = []
        bin_means = []
        for b in range(n_bins):
            start = b * bin_size
            end = start + bin_size if b < n_bins - 1 else len(sorted_idx)
            idx = sorted_idx[start:end]
            bin_centers.append(np.mean(uncertainties[qed_valid][idx]))
            bin_means.append(np.mean(props['qed'][qed_valid][idx]))
        ax2.plot(bin_centers, bin_means, 'r-o', linewidth=2, markersize=4, label='Binned mean')
        corr = np.corrcoef(uncertainties[qed_valid], props['qed'][qed_valid])[0, 1]
        ax2.set_title(f'QED vs Uncertainty (r={corr:.3f})')
    ax2.set_xlabel('Uncertainty')
    ax2.set_ylabel('QED')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. SA Score vs Uncertainty scatter
    ax3 = fig.add_subplot(gs[0, 2])
    sa_valid = np.isfinite(props['sa']) & finite_mask
    if sa_valid.sum() > 0:
        ax3.scatter(uncertainties[sa_valid], props['sa'][sa_valid],
                    alpha=0.1, s=5, c='#2C3E50')
        sorted_idx = np.argsort(uncertainties[sa_valid])
        bin_size = len(sorted_idx) // n_bins
        bin_centers = []
        bin_means = []
        for b in range(n_bins):
            start = b * bin_size
            end = start + bin_size if b < n_bins - 1 else len(sorted_idx)
            idx = sorted_idx[start:end]
            bin_centers.append(np.mean(uncertainties[sa_valid][idx]))
            bin_means.append(np.mean(props['sa'][sa_valid][idx]))
        ax3.plot(bin_centers, bin_means, 'r-o', linewidth=2, markersize=4, label='Binned mean')
        corr = np.corrcoef(uncertainties[sa_valid], props['sa'][sa_valid])[0, 1]
        ax3.set_title(f'SA Score vs Uncertainty (r={corr:.3f})')
    ax3.set_xlabel('Uncertainty')
    ax3.set_ylabel('SA Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. LogP vs Uncertainty
    ax4 = fig.add_subplot(gs[1, 0])
    logp_valid = np.isfinite(props['logp']) & finite_mask
    if logp_valid.sum() > 0:
        ax4.scatter(uncertainties[logp_valid], props['logp'][logp_valid],
                    alpha=0.1, s=5, c='#2C3E50')
        sorted_idx = np.argsort(uncertainties[logp_valid])
        bin_size = len(sorted_idx) // n_bins
        bin_centers = []
        bin_means = []
        for b in range(n_bins):
            start = b * bin_size
            end = start + bin_size if b < n_bins - 1 else len(sorted_idx)
            idx = sorted_idx[start:end]
            bin_centers.append(np.mean(uncertainties[logp_valid][idx]))
            bin_means.append(np.mean(props['logp'][logp_valid][idx]))
        ax4.plot(bin_centers, bin_means, 'r-o', linewidth=2, markersize=4, label='Binned mean')
        corr = np.corrcoef(uncertainties[logp_valid], props['logp'][logp_valid])[0, 1]
        ax4.set_title(f'LogP vs Uncertainty (r={corr:.3f})')
    ax4.set_xlabel('Uncertainty')
    ax4.set_ylabel('LogP')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. MW vs Uncertainty
    ax5 = fig.add_subplot(gs[1, 1])
    mw_valid = np.isfinite(props['mw']) & finite_mask
    if mw_valid.sum() > 0:
        ax5.scatter(uncertainties[mw_valid], props['mw'][mw_valid],
                    alpha=0.1, s=5, c='#2C3E50')
        sorted_idx = np.argsort(uncertainties[mw_valid])
        bin_size = len(sorted_idx) // n_bins
        bin_centers = []
        bin_means = []
        for b in range(n_bins):
            start = b * bin_size
            end = start + bin_size if b < n_bins - 1 else len(sorted_idx)
            idx = sorted_idx[start:end]
            bin_centers.append(np.mean(uncertainties[mw_valid][idx]))
            bin_means.append(np.mean(props['mw'][mw_valid][idx]))
        ax5.plot(bin_centers, bin_means, 'r-o', linewidth=2, markersize=4, label='Binned mean')
        corr = np.corrcoef(uncertainties[mw_valid], props['mw'][mw_valid])[0, 1]
        ax5.set_title(f'MW vs Uncertainty (r={corr:.3f})')
    ax5.set_xlabel('Uncertainty')
    ax5.set_ylabel('Molecular Weight')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Cumulative quality curves
    ax6 = fig.add_subplot(gs[1, 2])
    sorted_idx_global = np.argsort(uncertainties)
    fractions = np.linspace(0.1, 1.0, 50)

    for pname, plabel, color in [
        ('qed', 'Mean QED', '#27AE60'),
        ('sa', 'Mean SA (inverted)', '#E74C3C'),
    ]:
        cum_vals = []
        for frac in fractions:
            n_keep = int(frac * len(smiles))
            idx = sorted_idx_global[:n_keep]
            vals = props[pname][idx]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                if pname == 'sa':
                    cum_vals.append(1.0 / np.mean(vals))  # invert SA so higher=better
                else:
                    cum_vals.append(np.mean(vals))
            else:
                cum_vals.append(np.nan)
        ax6.plot(fractions * 100, cum_vals, '-', color=color, linewidth=2, label=plabel)

    ax6.set_xlabel('% of molecules kept (by lowest uncertainty)')
    ax6.set_ylabel('Quality Metric')
    ax6.set_title('Cumulative Quality vs Filtering')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. FCD at different filtering levels (from saved data)
    ax7 = fig.add_subplot(gs[2, 0])
    # We'll compute FCD if possible, otherwise skip
    ax7.text(0.5, 0.5, 'See filtering_results plot\nfor FCD comparison',
             ha='center', va='center', transform=ax7.transAxes, fontsize=12)
    ax7.set_title('FCD (see separate plot)')

    # 8. Uncertainty rank correlation
    ax8 = fig.add_subplot(gs[2, 1])
    # Show rank of uncertainty vs rank of QED
    if qed_valid.sum() > 100:
        from scipy import stats
        unc_rank = stats.rankdata(uncertainties[qed_valid])
        qed_rank = stats.rankdata(props['qed'][qed_valid])
        spearman_r, spearman_p = stats.spearmanr(uncertainties[qed_valid], props['qed'][qed_valid])
        ax8.scatter(unc_rank / len(unc_rank), qed_rank / len(qed_rank),
                    alpha=0.05, s=3, c='#2C3E50')
        ax8.set_xlabel('Uncertainty Rank (percentile)')
        ax8.set_ylabel('QED Rank (percentile)')
        ax8.set_title(f'Rank Correlation\n(Spearman ρ={spearman_r:.3f}, p={spearman_p:.2e})')
        ax8.grid(True, alpha=0.3)

    # 9. Uncertainty calibration: binned reliability
    ax9 = fig.add_subplot(gs[2, 2])
    # For each uncertainty bin, what fraction of posterior samples produce valid molecules?
    # (We don't have this info from saved results, so show uncertainty vs novelty)
    if len(smiles) > 100:
        # Check novelty per uncertainty bin
        n_bins_cal = 10
        sorted_idx_cal = np.argsort(uncertainties)
        bin_size_cal = len(sorted_idx_cal) // n_bins_cal
        novelty_rates = []
        unc_medians = []

        for b in range(n_bins_cal):
            start = b * bin_size_cal
            end = start + bin_size_cal if b < n_bins_cal - 1 else len(sorted_idx_cal)
            idx = sorted_idx_cal[start:end]
            unc_medians.append(np.median(uncertainties[idx]))

            # Count unique SMILES in this bin
            bin_smiles = [str(smiles[i]) for i in idx if str(smiles[i]) not in ('None', '')]
            unique_in_bin = len(set(bin_smiles))
            novelty_rates.append(unique_in_bin / len(bin_smiles) if len(bin_smiles) > 0 else 0)

        ax9.bar(range(n_bins_cal), novelty_rates, color='#9B59B6', alpha=0.7)
        ax9.set_xticks(range(n_bins_cal))
        ax9.set_xticklabels([f'{i*10}-{(i+1)*10}%' for i in range(n_bins_cal)],
                            rotation=45, ha='right', fontsize=8)
        ax9.set_xlabel('Uncertainty Percentile')
        ax9.set_ylabel('Uniqueness Rate')
        ax9.set_title('Uniqueness by Uncertainty Bin')
        ax9.grid(True, alpha=0.3)

    plt.suptitle('Comprehensive Uncertainty Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'uncertainty_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive analysis saved to: {path}")


def molecule_grid_by_uncertainty(smiles, uncertainties, output_dir, n_examples=16):
    """
    Create a grid showing molecules at different uncertainty levels.
    2x2 quadrants: lowest uncertainty, 25th percentile, 75th percentile, highest uncertainty.
    """
    finite_mask = np.isfinite(uncertainties)
    valid_idx = np.where(finite_mask)[0]
    sorted_idx = valid_idx[np.argsort(uncertainties[valid_idx])]

    n = len(sorted_idx)
    n_per_group = min(n_examples, 8)

    groups = {
        'Lowest Uncertainty\n(Most Confident)': sorted_idx[:n_per_group],
        '25th Percentile': sorted_idx[n//4 - n_per_group//2 : n//4 + n_per_group//2],
        '75th Percentile': sorted_idx[3*n//4 - n_per_group//2 : 3*n//4 + n_per_group//2],
        'Highest Uncertainty\n(Least Confident)': sorted_idx[-n_per_group:],
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    for ax, (title, idx_group) in zip(axes.flatten(), groups.items()):
        mols = []
        legends = []
        for i in idx_group:
            smi = str(smiles[i])
            mol = Chem.MolFromSmiles(smi) if smi not in ('None', '') else None
            if mol is not None:
                mols.append(mol)
                legends.append(f'U={uncertainties[i]:.1f}')

        if len(mols) > 0:
            n_cols = min(4, len(mols))
            n_rows = (len(mols) + n_cols - 1) // n_cols
            img = Draw.MolsToGridImage(mols, molsPerRow=n_cols, subImgSize=(300, 250),
                                       legends=legends)
            ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle('Molecules by Uncertainty Level', fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'molecule_grid_uncertainty.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Molecule grid saved to: {path}")


def summary_statistics(smiles, uncertainties, output_dir):
    """Print and save summary statistics."""
    props = compute_mol_properties(smiles)
    finite_mask = np.isfinite(uncertainties)

    # Split into low/high uncertainty halves
    sorted_idx = np.argsort(uncertainties)
    n = len(sorted_idx)
    low_unc_idx = sorted_idx[:n//2]
    high_unc_idx = sorted_idx[n//2:]

    lines = []
    lines.append("=" * 70)
    lines.append("UNCERTAINTY ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"\nTotal molecules: {len(smiles)}")
    lines.append(f"Uncertainty range: [{uncertainties.min():.2f}, {uncertainties.max():.2f}]")
    lines.append(f"Uncertainty mean: {uncertainties.mean():.2f} (std: {uncertainties.std():.2f})")

    lines.append(f"\n{'Property':<20} {'Low Unc (50%)':<20} {'High Unc (50%)':<20} {'Difference':<15}")
    lines.append("-" * 70)

    for pname, plabel in [('qed', 'QED'), ('sa', 'SA Score'),
                          ('logp', 'LogP'), ('mw', 'Mol. Weight'),
                          ('num_rings', 'Ring Count'), ('num_atoms', 'Atom Count')]:
        low_vals = props[pname][low_unc_idx]
        high_vals = props[pname][high_unc_idx]
        low_vals = low_vals[np.isfinite(low_vals)]
        high_vals = high_vals[np.isfinite(high_vals)]

        if len(low_vals) > 0 and len(high_vals) > 0:
            low_mean = np.mean(low_vals)
            high_mean = np.mean(high_vals)
            diff = low_mean - high_mean
            lines.append(f"{plabel:<20} {low_mean:<20.4f} {high_mean:<20.4f} {diff:+.4f}")

    # Correlation with uncertainty
    lines.append(f"\n{'Property':<20} {'Pearson r':<15} {'Spearman ρ':<15}")
    lines.append("-" * 50)

    try:
        from scipy import stats
        for pname, plabel in [('qed', 'QED'), ('sa', 'SA Score'),
                              ('logp', 'LogP'), ('mw', 'Mol. Weight')]:
            valid = np.isfinite(props[pname]) & finite_mask
            if valid.sum() > 10:
                pearson_r = np.corrcoef(uncertainties[valid], props[pname][valid])[0, 1]
                spearman_r, _ = stats.spearmanr(uncertainties[valid], props[pname][valid])
                lines.append(f"{plabel:<20} {pearson_r:<15.4f} {spearman_r:<15.4f}")
    except ImportError:
        lines.append("(scipy not available for Spearman correlation)")

    summary = '\n'.join(lines)
    print(summary)

    path = os.path.join(output_dir, 'uncertainty_summary.txt')
    with open(path, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--dataset', type=str, default='ZINC250k')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as results_dir)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'analysis')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Comprehensive Uncertainty Analysis")
    print("=" * 70)

    # Load results
    smiles, uncertainties, raw_data = load_results(args.results_dir, args.dataset)
    print(f"Loaded {len(smiles)} molecules from {args.results_dir}")
    print(f"Uncertainty range: [{uncertainties.min():.2f}, {uncertainties.max():.2f}]")

    # Load reference SMILES for precision/recall
    try:
        from utils.mol_utils import load_smiles, canonicalize_smiles
        train_smiles, test_smiles = load_smiles(args.dataset)
        train_smiles = canonicalize_smiles(train_smiles)
        test_smiles = canonicalize_smiles(test_smiles)
        print(f"Loaded reference SMILES: train={len(train_smiles)}, test={len(test_smiles)}")
    except Exception as e:
        print(f"Could not load reference SMILES: {e}")
        train_smiles = None
        test_smiles = None

    # 1. Summary statistics
    print("\n" + "=" * 70)
    summary_statistics(smiles, uncertainties, args.output_dir)

    # 2. Comprehensive analysis plot
    print("\n" + "=" * 70)
    print("Creating comprehensive analysis plots...")
    uncertainty_distribution_analysis(smiles, uncertainties, args.output_dir)

    # 3. Property vs uncertainty
    print("Creating property correlation plots...")
    uncertainty_property_correlation(smiles, uncertainties, args.output_dir)

    # 4. Precision/recall
    if train_smiles is not None:
        print("Creating precision/recall curves...")
        precision_recall_curves(smiles, uncertainties, train_smiles, test_smiles, args.output_dir)

    # 5. Molecule grid
    print("Creating molecule grid...")
    molecule_grid_by_uncertainty(smiles, uncertainties, args.output_dir)

    print("\n" + "=" * 70)
    print(f"All analysis outputs saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
