"""
Analyze X-network vs Adj-network Uncertainty

Compares different uncertainty aggregation strategies for filtering and
analyzes what kind of molecules each network's uncertainty selects for.

Usage:
    python analyze_network_uncertainties.py --results_dir ./results_steps_nocorrect --steps 100
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

sys.path.insert(0, '.')


def load_results(results_dir, dataset, steps):
    path = os.path.join(results_dir, f'results_{dataset}_{steps}steps.npz')
    data = np.load(path, allow_pickle=True)
    return data


def compute_props(smiles_list):
    """Compute per-molecule properties."""
    props = {'valid': [], 'qed': [], 'sa': [], 'logp': [], 'mw': [],
             'num_rings': [], 'num_atoms': []}

    sa_scorer = None
    try:
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        import sascorer
        sa_scorer = sascorer
    except Exception:
        pass

    for smi in smiles_list:
        mol = None
        if smi is not None and str(smi) not in ('None', ''):
            mol = Chem.MolFromSmiles(str(smi))

        if mol is None:
            props['valid'].append(0)
            for k in ['qed', 'sa', 'logp', 'mw', 'num_rings', 'num_atoms']:
                props[k].append(np.nan)
        else:
            props['valid'].append(1)
            props['qed'].append(QED.qed(mol))
            props['sa'].append(sa_scorer.calculateScore(mol) if sa_scorer else np.nan)
            props['logp'].append(Descriptors.MolLogP(mol))
            props['mw'].append(Descriptors.MolWt(mol))
            props['num_rings'].append(Descriptors.RingCount(mol))
            props['num_atoms'].append(mol.GetNumHeavyAtoms())

    return {k: np.array(v) for k, v in props.items()}


def compute_filtering_metrics(smiles_list, uncertainties, fractions):
    """Compute validity at different filtering levels for a given uncertainty."""
    n_total = len(smiles_list)

    # Handle inf: replace with large finite value so they sort last
    unc_clean = np.copy(uncertainties)
    unc_clean[~np.isfinite(unc_clean)] = 1e10

    sorted_idx = np.argsort(unc_clean)

    results = {'n': [], 'validity': [], 'mean_qed': [], 'mean_mw': [],
               'mean_sa': [], 'fcd': []}

    props = compute_props(smiles_list)

    for frac in fractions:
        n_keep = int(frac * n_total)
        kept_idx = sorted_idx[:n_keep]

        validity = props['valid'][kept_idx].mean()
        qed_vals = props['qed'][kept_idx]
        qed_vals = qed_vals[np.isfinite(qed_vals)]
        mw_vals = props['mw'][kept_idx]
        mw_vals = mw_vals[np.isfinite(mw_vals)]
        sa_vals = props['sa'][kept_idx]
        sa_vals = sa_vals[np.isfinite(sa_vals)]

        results['n'].append(n_keep)
        results['validity'].append(validity)
        results['mean_qed'].append(np.mean(qed_vals) if len(qed_vals) > 0 else np.nan)
        results['mean_mw'].append(np.mean(mw_vals) if len(mw_vals) > 0 else np.nan)
        results['mean_sa'].append(np.mean(sa_vals) if len(sa_vals) > 0 else np.nan)

        # FCD
        valid_smiles = [str(smiles_list[i]) for i in kept_idx
                        if props['valid'][i] == 1]
        if len(valid_smiles) >= 10:
            try:
                from utils.mol_utils import load_smiles, canonicalize_smiles
                _, test_smiles = load_smiles('ZINC250k')
                test_smiles = canonicalize_smiles(test_smiles)
                from fcd_torch import FCD
                fcd_calc = FCD(device='cpu', n_jobs=1)
                fcd_val = fcd_calc(valid_smiles, test_smiles[:len(valid_smiles)*2])
                results['fcd'].append(fcd_val)
            except Exception:
                results['fcd'].append(np.nan)
        else:
            results['fcd'].append(np.nan)

    return results


def plot_aggregation_comparison(smiles, unc_combined, unc_x, unc_adj, output_dir, steps):
    """Compare different uncertainty aggregation strategies for filtering."""

    # Build strategies
    unc_mean = (unc_x + unc_adj) / 2
    unc_sum = unc_x + unc_adj

    # For max: need to handle that lower uncertainty = better,
    # so "max uncertainty" = the network that is LEAST confident
    unc_max = np.maximum(unc_x, unc_adj)

    strategies = {
        'Combined (joint)': unc_combined,
        'X-only': unc_x,
        'Adj-only': unc_adj,
        'Mean(X, Adj)': unc_mean,
        'Max(X, Adj)': unc_max,
    }

    # Compute validity for each molecule
    valid_flags = np.zeros(len(smiles))
    for i, smi in enumerate(smiles):
        if smi is not None and str(smi) not in ('None', ''):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                valid_flags[i] = 1.0

    overall_validity = valid_flags.mean()
    n_total = len(smiles)
    fracs = np.linspace(0.1, 1.0, 30)

    colors = {
        'Combined (joint)': '#2C3E50',
        'X-only': '#E74C3C',
        'Adj-only': '#3498DB',
        'Mean(X, Adj)': '#27AE60',
        'Max(X, Adj)': '#9B59B6',
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Validity vs filtering fraction
    ax = axes[0]
    for name, unc in strategies.items():
        unc_clean = np.copy(unc)
        unc_clean[~np.isfinite(unc_clean)] = 1e10
        sorted_idx = np.argsort(unc_clean)

        validity_curve = []
        for frac in fracs:
            n_keep = int(frac * n_total)
            kept = sorted_idx[:n_keep]
            validity_curve.append(valid_flags[kept].mean())

        ax.plot(fracs * 100, validity_curve, '-', color=colors[name],
                linewidth=2, label=name)

    ax.axhline(overall_validity, color='gray', linestyle='--', linewidth=1,
               label=f'Overall: {overall_validity:.1%}')
    ax.set_xlabel('% of molecules kept')
    ax.set_ylabel('Validity')
    ax.set_title('Validity by Aggregation Strategy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # 2. Mean QED vs filtering fraction
    ax = axes[1]
    props = compute_props(smiles)

    for name, unc in strategies.items():
        unc_clean = np.copy(unc)
        unc_clean[~np.isfinite(unc_clean)] = 1e10
        sorted_idx = np.argsort(unc_clean)

        qed_curve = []
        for frac in fracs:
            n_keep = int(frac * n_total)
            kept = sorted_idx[:n_keep]
            qed_vals = props['qed'][kept]
            qed_vals = qed_vals[np.isfinite(qed_vals)]
            qed_curve.append(np.mean(qed_vals) if len(qed_vals) > 0 else np.nan)

        ax.plot(fracs * 100, qed_curve, '-', color=colors[name],
                linewidth=2, label=name)

    overall_qed = np.nanmean(props['qed'])
    ax.axhline(overall_qed, color='gray', linestyle='--', linewidth=1,
               label=f'Overall: {overall_qed:.3f}')
    ax.set_xlabel('% of molecules kept')
    ax.set_ylabel('Mean QED')
    ax.set_title('QED by Aggregation Strategy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Mean MW vs filtering fraction
    ax = axes[2]
    for name, unc in strategies.items():
        unc_clean = np.copy(unc)
        unc_clean[~np.isfinite(unc_clean)] = 1e10
        sorted_idx = np.argsort(unc_clean)

        mw_curve = []
        for frac in fracs:
            n_keep = int(frac * n_total)
            kept = sorted_idx[:n_keep]
            mw_vals = props['mw'][kept]
            mw_vals = mw_vals[np.isfinite(mw_vals)]
            mw_curve.append(np.mean(mw_vals) if len(mw_vals) > 0 else np.nan)

        ax.plot(fracs * 100, mw_curve, '-', color=colors[name],
                linewidth=2, label=name)

    overall_mw = np.nanmean(props['mw'])
    ax.axhline(overall_mw, color='gray', linestyle='--', linewidth=1,
               label=f'Overall: {overall_mw:.0f}')
    ax.set_xlabel('% of molecules kept')
    ax.set_ylabel('Mean Molecular Weight')
    ax.set_title('MW by Aggregation Strategy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Uncertainty Aggregation Comparison ({steps} steps, no correction)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'aggregation_comparison_{steps}steps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Aggregation comparison saved to: {path}")


def plot_network_molecule_comparison(smiles, unc_x, unc_adj, output_dir, steps):
    """
    Analyze what kind of molecules each network selects for.
    Compare properties of molecules selected by x-only vs adj-only filtering.
    """
    from rdkit.Chem import Draw

    props = compute_props(smiles)
    n = len(smiles)

    # Clean up infinities
    unc_x_clean = np.copy(unc_x)
    unc_x_clean[~np.isfinite(unc_x_clean)] = 1e10
    unc_adj_clean = np.copy(unc_adj)
    unc_adj_clean[~np.isfinite(unc_adj_clean)] = 1e10

    # Select top 30% by each strategy
    frac = 0.3
    n_keep = int(frac * n)

    x_top = set(np.argsort(unc_x_clean)[:n_keep])
    adj_top = set(np.argsort(unc_adj_clean)[:n_keep])

    x_only_set = x_top - adj_top      # Selected by X but not Adj
    adj_only_set = adj_top - x_top     # Selected by Adj but not X
    both_set = x_top & adj_top         # Selected by both
    neither_set = set(range(n)) - x_top - adj_top  # Selected by neither

    groups = {
        f'X-only selected\n(n={len(x_only_set)})': list(x_only_set),
        f'Adj-only selected\n(n={len(adj_only_set)})': list(adj_only_set),
        f'Both selected\n(n={len(both_set)})': list(both_set),
        f'Neither selected\n(n={len(neither_set)})': list(neither_set),
    }

    # --- Figure 1: Property comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    prop_list = [
        ('valid', 'Validity'),
        ('qed', 'QED'),
        ('sa', 'SA Score'),
        ('logp', 'LogP'),
        ('mw', 'Molecular Weight'),
        ('num_atoms', 'Heavy Atom Count'),
    ]

    group_names = list(groups.keys())
    group_colors = ['#E74C3C', '#3498DB', '#27AE60', '#95A5A6']
    x_pos = np.arange(len(group_names))

    for ax, (pname, plabel) in zip(axes.flatten(), prop_list):
        means = []
        stds = []
        for gname, gidx in groups.items():
            if len(gidx) == 0:
                means.append(0)
                stds.append(0)
                continue
            vals = props[pname][gidx]
            vals = vals[np.isfinite(vals)]
            means.append(np.mean(vals) if len(vals) > 0 else 0)
            stds.append(np.std(vals) if len(vals) > 0 else 0)

        bars = ax.bar(x_pos, means, yerr=stds, color=group_colors, alpha=0.7, capsize=3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([g.split('\n')[0] for g in group_names],
                           rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(plabel)
        ax.set_title(plabel)
        ax.grid(True, alpha=0.3, axis='y')

        if pname == 'valid':
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    plt.suptitle(f'Molecular Properties by Selection Group (top 30%, {steps} steps)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'network_property_comparison_{steps}steps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Property comparison saved to: {path}")

    # --- Figure 2: Molecule grid for each group ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 18))
    n_show = 8

    for ax, (gname, gidx), color in zip(axes.flatten(), groups.items(), group_colors):
        if len(gidx) == 0:
            ax.text(0.5, 0.5, 'Empty', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(gname)
            ax.axis('off')
            continue

        gidx = np.array(gidx)
        # Pick valid molecules from the middle of the group
        valid_in_group = gidx[props['valid'][gidx] == 1]

        if len(valid_in_group) == 0:
            ax.text(0.5, 0.5, 'No valid molecules', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(gname)
            ax.axis('off')
            continue

        # Sample representative molecules
        np.random.seed(42)
        sample_idx = valid_in_group[np.random.choice(len(valid_in_group),
                                                      min(n_show, len(valid_in_group)),
                                                      replace=False)]

        mols = []
        legends = []
        for i in sample_idx:
            smi = str(smiles[i])
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                legends.append(f'Ux={unc_x[i]:.0f}\nUa={unc_adj[i]:.0f}')

        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(300, 250),
                                       legends=legends)
            ax.imshow(img)

        # Stats
        v = props['valid'][gidx].mean()
        q = np.nanmean(props['qed'][gidx])
        m = np.nanmean(props['mw'][gidx])
        ax.set_title(f'{gname}\nValid={v:.0%} | QED={q:.3f} | MW={m:.0f}',
                     fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle(f'Molecules by Selection Group (top 30% filtering, {steps} steps)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'network_molecule_grid_{steps}steps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Molecule grid saved to: {path}")


def compute_fcd_for_subset(smiles_list, valid_flags, indices, test_smiles, fcd_calc):
    """Compute FCD for a subset of molecules."""
    valid_smiles = [str(smiles_list[i]) for i in indices if valid_flags[i] == 1]
    if len(valid_smiles) < 10:
        return np.nan
    try:
        return fcd_calc(valid_smiles, test_smiles[:len(valid_smiles)*2])
    except Exception:
        return np.nan


def plot_x_adj_filtering_table(smiles, unc_combined, unc_x, unc_adj, output_dir, steps, skip_fcd=False):
    """
    Comprehensive table: for each strategy, show validity AND FCD at key filtering levels.
    """
    strategies = {
        'Combined': unc_combined,
        'X-only': unc_x,
        'Adj-only': unc_adj,
        'Mean(X,Adj)': (unc_x + unc_adj) / 2,
        'Max(X,Adj)': np.maximum(unc_x, unc_adj),
    }

    valid_flags = np.zeros(len(smiles))
    for i, smi in enumerate(smiles):
        if smi is not None and str(smi) not in ('None', ''):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                valid_flags[i] = 1.0

    # Load test SMILES and FCD calculator
    has_fcd = False
    test_smiles = None
    fcd_calc = None
    if not skip_fcd:
        try:
            from utils.mol_utils import load_smiles, canonicalize_smiles
            from fcd_torch import FCD
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            _, test_smiles = load_smiles('ZINC250k')
            test_smiles = canonicalize_smiles(test_smiles)
            fcd_calc = FCD(device=device, n_jobs=1)
            has_fcd = True
            print(f"  FCD initialized on {device}")
        except Exception as e:
            print(f"  FCD not available: {e}")

    n_total = len(smiles)
    fracs = [0.3, 0.5, 0.7, 0.9, 1.0]

    # --- Validity table ---
    lines = []
    lines.append(f"VALIDITY by Strategy ({steps} steps, no correction, n={n_total})")
    lines.append(f"{'Strategy':<16} | " + " | ".join(f"{'Keep '+str(int(f*100))+'%':>10}" for f in fracs))
    lines.append("-" * (18 + 13 * len(fracs)))

    random_vals = [f"{valid_flags.mean():>9.1%}" for _ in fracs]
    lines.append(f"{'Random':<16} | " + " | ".join(random_vals))
    lines.append("-" * (18 + 13 * len(fracs)))

    for name, unc in strategies.items():
        unc_clean = np.copy(unc)
        unc_clean[~np.isfinite(unc_clean)] = 1e10
        sorted_idx = np.argsort(unc_clean)
        vals = []
        for frac in fracs:
            n_keep = int(frac * n_total)
            kept = sorted_idx[:n_keep]
            vals.append(f"{valid_flags[kept].mean():>9.1%}")
        lines.append(f"{name:<16} | " + " | ".join(vals))

    # --- FCD table ---
    if has_fcd:
        lines.append("")
        lines.append(f"FCD by Strategy ({steps} steps, no correction, n={n_total})")
        lines.append(f"{'Strategy':<16} | " + " | ".join(f"{'Keep '+str(int(f*100))+'%':>10}" for f in fracs))
        lines.append("-" * (18 + 13 * len(fracs)))

        # Random FCD (average over 3 runs)
        random_fcds = []
        for frac in fracs:
            n_keep = int(frac * n_total)
            fcd_runs = []
            for seed in range(3):
                np.random.seed(seed + 42)
                rand_idx = np.random.choice(n_total, n_keep, replace=False)
                fcd_val = compute_fcd_for_subset(smiles, valid_flags, rand_idx, test_smiles, fcd_calc)
                if np.isfinite(fcd_val):
                    fcd_runs.append(fcd_val)
            random_fcds.append(np.mean(fcd_runs) if fcd_runs else np.nan)
        lines.append(f"{'Random':<16} | " + " | ".join(f"{v:>10.2f}" if np.isfinite(v) else f"{'N/A':>10}" for v in random_fcds))
        lines.append("-" * (18 + 13 * len(fracs)))

        for name, unc in strategies.items():
            unc_clean = np.copy(unc)
            unc_clean[~np.isfinite(unc_clean)] = 1e10
            sorted_idx = np.argsort(unc_clean)
            vals = []
            for frac in fracs:
                n_keep = int(frac * n_total)
                kept = sorted_idx[:n_keep]
                fcd_val = compute_fcd_for_subset(smiles, valid_flags, kept, test_smiles, fcd_calc)
                vals.append(f"{fcd_val:>10.2f}" if np.isfinite(fcd_val) else f"{'N/A':>10}")
            lines.append(f"{name:<16} | " + " | ".join(vals))

    table = "\n".join(lines)
    print(table)

    path = os.path.join(output_dir, f'strategy_table_{steps}steps.txt')
    with open(path, 'w') as f:
        f.write(table)
    print(f"\nTable saved to: {path}")

    return table


def plot_strategy_precision_recall(smiles, unc_combined, unc_x, unc_adj, output_dir, steps, skip_fcd=False):
    """
    Precision/recall and FCD curves for all strategies on same axes.
    """
    strategies = {
        'Combined': unc_combined,
        'X-only': unc_x,
        'Adj-only': unc_adj,
        'Mean(X,Adj)': (unc_x + unc_adj) / 2,
        'Max(X,Adj)': np.maximum(unc_x, unc_adj),
    }

    colors = {
        'Combined': '#2C3E50',
        'X-only': '#E74C3C',
        'Adj-only': '#3498DB',
        'Mean(X,Adj)': '#27AE60',
        'Max(X,Adj)': '#9B59B6',
    }

    valid_flags = np.zeros(len(smiles))
    for i, smi in enumerate(smiles):
        if smi is not None and str(smi) not in ('None', ''):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                valid_flags[i] = 1.0

    n_total = len(smiles)
    n_valid_total = valid_flags.sum()
    overall_validity = n_valid_total / n_total

    # Load FCD
    has_fcd = False
    if not skip_fcd:
        try:
            from utils.mol_utils import load_smiles, canonicalize_smiles
            from fcd_torch import FCD
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            _, test_smiles = load_smiles('ZINC250k')
            test_smiles = canonicalize_smiles(test_smiles)
            fcd_calc = FCD(device=device, n_jobs=1)
            has_fcd = True
        except Exception:
            pass

    fracs = np.linspace(0.1, 1.0, 20)
    # Fewer points for FCD (expensive)
    fcd_fracs = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    fig, axes = plt.subplots(1, 3 if has_fcd else 2, figsize=(7 * (3 if has_fcd else 2), 6))

    # --- Panel 1: Precision-Recall ---
    ax = axes[0]
    for name, unc in strategies.items():
        unc_clean = np.copy(unc)
        unc_clean[~np.isfinite(unc_clean)] = 1e10
        sorted_idx = np.argsort(unc_clean)

        precisions = []
        recalls = []
        for frac in fracs:
            n_keep = int(frac * n_total)
            kept = sorted_idx[:n_keep]
            tp = valid_flags[kept].sum()
            precisions.append(tp / n_keep if n_keep > 0 else 0)
            recalls.append(tp / n_valid_total if n_valid_total > 0 else 0)

        ax.plot(recalls, precisions, '-', color=colors[name], linewidth=2, label=name)

    ax.plot([0, 1], [overall_validity, overall_validity], 'k--', linewidth=1,
            alpha=0.5, label=f'Random ({overall_validity:.1%})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision (Validity)')
    ax.set_title('Precision-Recall by Strategy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(max(0, overall_validity - 0.15), 1.02)

    # --- Panel 2: Validity vs % kept ---
    ax = axes[1]
    for name, unc in strategies.items():
        unc_clean = np.copy(unc)
        unc_clean[~np.isfinite(unc_clean)] = 1e10
        sorted_idx = np.argsort(unc_clean)

        validity_curve = []
        for frac in fracs:
            n_keep = int(frac * n_total)
            kept = sorted_idx[:n_keep]
            validity_curve.append(valid_flags[kept].mean())

        ax.plot(fracs * 100, validity_curve, '-', color=colors[name], linewidth=2, label=name)

    ax.axhline(overall_validity, color='k', linestyle='--', linewidth=1, alpha=0.5,
               label=f'Random ({overall_validity:.1%})')
    ax.set_xlabel('% of molecules kept')
    ax.set_ylabel('Validity')
    ax.set_title('Validity vs Filtering')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(max(0, overall_validity - 0.15), 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # --- Panel 3: FCD vs % kept ---
    if has_fcd:
        ax = axes[2]

        # Random FCD baseline
        random_fcds = []
        for frac in fcd_fracs:
            n_keep = int(frac * n_total)
            fcd_runs = []
            for seed in range(3):
                np.random.seed(seed + 42)
                rand_idx = np.random.choice(n_total, n_keep, replace=False)
                fcd_val = compute_fcd_for_subset(smiles, valid_flags, rand_idx, test_smiles, fcd_calc)
                if np.isfinite(fcd_val):
                    fcd_runs.append(fcd_val)
            random_fcds.append(np.mean(fcd_runs) if fcd_runs else np.nan)
        ax.plot(fcd_fracs * 100, random_fcds, 'k--', linewidth=1.5, alpha=0.7, label='Random')

        for name, unc in strategies.items():
            unc_clean = np.copy(unc)
            unc_clean[~np.isfinite(unc_clean)] = 1e10
            sorted_idx = np.argsort(unc_clean)

            fcd_curve = []
            for frac in fcd_fracs:
                n_keep = int(frac * n_total)
                kept = sorted_idx[:n_keep]
                fcd_val = compute_fcd_for_subset(smiles, valid_flags, kept, test_smiles, fcd_calc)
                fcd_curve.append(fcd_val)

            ax.plot(fcd_fracs * 100, fcd_curve, '-o', color=colors[name],
                    linewidth=2, markersize=4, label=name)

        ax.set_xlabel('% of molecules kept')
        ax.set_ylabel('FCD (lower is better)')
        ax.set_title('FCD vs Filtering')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Strategy Comparison: Precision/Recall & FCD ({steps} steps, no correction)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'strategy_pr_fcd_{steps}steps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Strategy PR/FCD plot saved to: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./results_steps_nocorrect')
    parser.add_argument('--dataset', type=str, default='ZINC250k')
    parser.add_argument('--steps', type=str, default='100',
                        help='Comma-separated step counts to analyze')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--skip_fcd', action='store_true',
                        help='Skip FCD computation (for fast runs on login nodes)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'network_analysis')
    os.makedirs(args.output_dir, exist_ok=True)

    step_list = [int(s) for s in args.steps.split(',')]

    for steps in step_list:
        print(f"\n{'='*70}")
        print(f"Analyzing {steps} steps")
        print(f"{'='*70}")

        try:
            data = load_results(args.results_dir, args.dataset, steps)
        except FileNotFoundError:
            print(f"  Results not found for {steps} steps, skipping")
            continue

        smiles = data['smiles']
        unc_combined = data['unc_combined']
        unc_x = data['unc_x']
        unc_adj = data['unc_adj']

        print(f"  Molecules: {len(smiles)}")
        print(f"  Combined unc range: [{np.nanmin(unc_combined):.1f}, {np.nanmax(unc_combined):.1f}]")
        print(f"  X-network unc range: [{np.nanmin(unc_x):.1f}, {np.nanmax(unc_x):.1f}]")
        print(f"  Adj-network unc range: [{np.nanmin(unc_adj):.1f}, {np.nanmax(unc_adj):.1f}]")

        # Correlation between x and adj uncertainties
        finite = np.isfinite(unc_x) & np.isfinite(unc_adj)
        if finite.sum() > 10:
            corr = np.corrcoef(unc_x[finite], unc_adj[finite])[0, 1]
            print(f"  X-Adj correlation: {corr:.3f}")

        plot_x_adj_filtering_table(smiles, unc_combined, unc_x, unc_adj,
                                   args.output_dir, steps, skip_fcd=args.skip_fcd)

        plot_strategy_precision_recall(smiles, unc_combined, unc_x, unc_adj,
                                       args.output_dir, steps, skip_fcd=args.skip_fcd)

        plot_aggregation_comparison(smiles, unc_combined, unc_x, unc_adj,
                                   args.output_dir, steps)

        plot_network_molecule_comparison(smiles, unc_x, unc_adj,
                                        args.output_dir, steps)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
