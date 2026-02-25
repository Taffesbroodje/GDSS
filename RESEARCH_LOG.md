# Research Log: Generative Uncertainty for Molecular Graph Diffusion

**Project**: Applying Jazbec et al. (2025) "Generative Uncertainty in Diffusion Models" to GDSS molecular generation on ZINC250k
**Author**: Pepijn Seij
**Started**: January 2025
**Last updated**: February 19, 2025

---

## Table of Contents

1. [Research Question](#1-research-question)
2. [Background](#2-background)
3. [Method](#3-method)
4. [Implementation Details](#4-implementation-details)
5. [Experiments & Results](#5-experiments--results)
6. [Analysis & Discussion](#6-analysis--discussion)
7. [Open Questions & Next Steps](#7-open-questions--next-steps)
8. [Timeline](#8-timeline)

---

## 1. Research Question

**Can generative uncertainty estimation, as proposed by Jazbec et al. (2025) for image diffusion models, be adapted to molecular graph diffusion to identify and filter low-quality generated molecules?**

Sub-questions:
- Does the Laplace approximation over score network parameters provide meaningful uncertainty in graph diffusion?
- Can uncertainty-based filtering improve the quality of generated molecular sets?
- How does the number of diffusion steps interact with uncertainty estimation?
- What is the role of the Langevin corrector in shaping uncertainty?

---

## 2. Background

### 2.1 Jazbec et al. (2025) — Generative Uncertainty in Diffusion Models

**Core idea**: Given a pretrained diffusion model with parameters theta, approximate the Bayesian posterior p(theta|D) via last-layer Laplace, then measure how much different posterior samples theta_m produce different outputs for the same input noise z.

**Algorithm** (Algorithm 1 from the paper):
1. Fix noise z
2. Generate baseline sample x_0 = g_{theta_hat}(z) using MAP parameters
3. For m = 1..M: sample theta_m ~ q(theta|D), generate x_m = g_{theta_m}(z)
4. Extract semantic features e_m = c_phi(x_m) using a pretrained encoder
5. Compute uncertainty u(z) = H(p(x|z,D)) as entropy of the feature distribution

**Key design choices in the paper**:
- Semantic encoder: CLIP (ViT-B/32) for images
- M = 5 posterior samples
- T = 50 diffusion steps (reduced from 250 for efficiency)
- sigma^2 = 0.001 observation noise
- Diagonal empirical Fisher, last-layer only
- Uses `laplace-torch` library

**Paper results** (ImageNet):
- UViT (10K filtered): FID 7.89 vs 9.45 random — clear improvement
- ADM: Similar improvements
- High-uncertainty images show visible artifacts; low-uncertainty ones are clean

**Important**: The paper only experiments with image generation. No molecular generation experiments exist. Our work is a novel application of this framework to the molecular domain.

### 2.2 GDSS — Score-Based Graph Diffusion

GDSS (Jo et al., ICML 2022) generates molecular graphs via a system of two coupled SDEs:
- **X-SDE** (VP-SDE): For node features (atom types), beta_min=0.1, beta_max=1.0
- **A-SDE** (VE-SDE): For adjacency matrix (bond types), beta_min=0.2, beta_max=1.0

The coupling is in the score networks: both ScoreNetworkX and ScoreNetworkA condition on the joint state (X, A).

**PC Sampler**: At each diffusion step, a Langevin corrector refines the marginal, then a reverse-diffusion predictor takes one step. The corrector is crucial at low step counts to fix discretization errors.

### 2.3 Key Differences: Images vs Molecules

| Aspect | Images (Jazbec) | Molecules (Ours) |
|--------|-----------------|-------------------|
| Output space | Continuous pixels | Discrete graphs (quantized) |
| Quality measure | Continuous (perceptual) | Binary (valid/invalid) + distributional |
| Semantic encoder | CLIP (400M image-text pairs) | ChemNet (6K bioassays) |
| Score network | Single U-Net (~500M params) | Two coupled networks (11K + 63K params) |
| Last layer params | Large | 747 (X) and 93 (A) |
| Correction | None needed | Langevin corrector enforces validity |

---

## 3. Method

### 3.1 Our Adaptation

We adapt Jazbec et al. to GDSS molecular generation:

1. **Laplace approximation**: Custom diagonal empirical Fisher over last layer (`final.linears.2`) of both ScoreNetworkX and ScoreNetworkA
2. **Semantic encoder**: ChemNet (512-dim embeddings from penultimate layer, same network used in FCD metric)
3. **Uncertainty metric**: Entropy of diagonal Gaussian fitted to ChemNet embeddings across M posterior samples
4. **Evaluation**: Uncertainty-based filtering — rank molecules by uncertainty, keep lowest-uncertainty subset, measure validity/FCD/uniqueness/novelty

### 3.2 Three Types of Uncertainty

We decompose uncertainty by network:
- **Combined**: Both theta_x and theta_a sampled simultaneously
- **X-only**: Only theta_x sampled, theta_a fixed at MAP
- **Adj-only**: Only theta_a sampled, theta_x fixed at MAP

### 3.3 Correction vs No-Correction

We test two GDSS sampling modes:
- **With correction**: Langevin corrector active (snr=0.2, scale_eps=0.9, n_steps=1) — standard GDSS
- **Without correction**: NoneCorrector — no Langevin refinement, more raw/noisy output

---

## 4. Implementation Details

### 4.1 Laplace Approximation (`laplace_gdss_full.py`)

- **Fisher computation**: Accumulate squared gradients of score matching loss (MSE between predicted and target scores) over training data batches
- **Posterior**: q(theta|D) = N(theta_hat, (H + prior_precision * I)^{-1}), where H = sum of squared gradients (diagonal)
- **Prior precision**: 1.0 (default)
- **Last layer sizes**: ScoreNetworkX: 747 params (6.5% of total), ScoreNetworkA: 93 params (0.15% of total)
- **Fisher normalization bug** (fixed Feb 17): Originally divided Fisher by n_data, making posterior ~30x too diffuse for X-network. Fixed by removing normalization. See Section 5.2.

### 4.2 Semantic Uncertainty (`chemnet_semantic_uncertainty.py`)

- **ChemNet embeddings**: 512-dim from `fcd_torch` library (penultimate layer of ChemNet trained on ~6K bioassays)
- **Invalid molecules**: Get zero-vector embeddings (increases variance = higher uncertainty, correct signal)
- **All-invalid**: Returns large finite penalty (H ≈ 1905.4)
- **Entropy formula**: H(p) = 0.5 * D * (1 + log(2pi)) + 0.5 * sum_d(log(var_d + sigma^2))

### 4.3 Evaluation Pipeline (`evaluate_uncertainty_steps.py`)

For each step count and correction mode:
1. Load pretrained GDSS model + pre-fitted Laplace state
2. For each batch of 200 molecules:
   - Fix noise z (seed-based)
   - Generate MAP sample + 10 posterior samples with same z
   - Compute ChemNet embeddings for each
   - Calculate per-molecule entropy as uncertainty
3. Aggregate results: validity, FCD, uniqueness, novelty at filter fractions [0.3, 0.4, ..., 1.0]

---

## 5. Experiments & Results

### 5.1 Comprehensive Step Sweep (Feb 17-18, Job 19584489)

**Setup**: 1000 molecules, 10 posterior samples, steps in {50, 100, 200, 500, 1000}, both correct and nocorrect.
**Output**: `results_comprehensive/`

#### Results Summary — WITH Correction

| Steps | Validity | FCD_all | FCD_50%_random | FCD_50%_combined | FCD_50%_adj |
|-------|----------|---------|----------------|-----------------|-------------|
| 50    | 100%     | 21.47   | 21.50          | 23.40           | 23.24       |
| 100   | 100%     | 16.53   | 16.93          | 19.00           | 18.66       |
| 200   | 100%     | 16.76   | 16.98          | 18.76           | 18.55       |
| 500   | 100%     | 16.61   | 17.06          | 19.06           | 18.40       |
| 1000  | 100%     | 16.88   | 17.24          | 19.41           | 18.71       |

**Key finding**: Uncertainty filtering makes FCD ~2 points WORSE than random at all step counts. Validity is 100% everywhere, so filtering cannot improve it.

#### Results Summary — WITHOUT Correction

| Steps | Validity | FCD_all | FCD_50%_random | FCD_50%_combined | FCD_50%_adj |
|-------|----------|---------|----------------|-----------------|-------------|
| 50    | 53.3%    | 20.63   | 22.35          | 24.11           | 21.44       |
| 100   | 85.8%    | 16.21   | 16.96          | 18.10           | 17.78       |
| 200   | 91.0%    | 16.88   | 17.32          | 17.92           | 18.35       |
| 500   | 93.4%    | 16.62   | 17.06          | 18.57           | 18.30       |
| 1000  | 96.0%    | 17.07   | 17.16          | 18.73           | 18.63       |

**Positive finding for validity**: Adj-only filtering without correction improves validity:
- 50 steps: Top 30% adj_only = 76.3% vs random 54.7% (+21.6pp)
- 100 steps: Top 30% adj_only = 83.3% vs random 86.3%
- 200 steps: Top 30% adj_only = 91.7% vs random 90.9%

#### Uncertainty Distributions

| Variant | Steps | Mean unc_combined | Std | Range |
|---------|-------|-------------------|-----|-------|
| Correct | 1000  | -351.2 | 39.7 | [-523, -246] |
| Nocorrect | 1000 | -373.8 | 252.2 | [-819, +1905] |
| Nocorrect | 50 | +7.4 | 1003.5 | [-882, +1905] |

Without correction, 11-216 molecules (depending on step count) hit the saturation value of 1905.4 (all posterior samples produce invalid molecules).

### 5.2 Fisher Normalization Bug (Fixed Feb 17)

**Bug**: `laplace_gdss_full.py` line 211 divided Fisher by n_data, turning the sum into an average. With prior_precision=1.0, the prior dominated the posterior, causing:
- X-network posterior std ≈ 0.95 (nearly prior) vs corrected ≈ 0.029
- 99.5% of molecules got unc_x = inf
- Only adj-only results were valid in pre-fix runs

**Fix**: Removed `self.H = self.H / n_data`. Old Laplace states backed up as `laplace_*_old_normalized.pt`.

### 5.3 Low-Step Experiments (Planned — Feb 19)

**Motivation**: At very low step counts (10, 20), generation quality should be poor, giving uncertainty filtering more room to improve results. This tests whether uncertainty is most useful in the low-quality regime.

**Job**: `~/jobs/eval_low_steps.job` — runs 10 and 20 steps, both correction modes, 1000 molecules each.

---

## 6. Analysis & Discussion

### 6.1 Why Uncertainty Filtering Hurts FCD

We identified five root causes, ordered by importance:

**1. Langevin correction eliminates the uncertainty signal**
With correction, validity is 100% at all step counts. There are no "bad" molecules for uncertainty to identify. The corrector runs at each diffusion step and effectively enforces chemical validity through iterative score-guided refinement. Without correction, uncertainty (especially adj_only) successfully predicts validity.

**2. ChemNet is not an adequate replacement for CLIP**
CLIP captures rich perceptual quality from 400M image-text pairs. ChemNet is trained on ~6K bioassays and captures chemical/biological properties, not "generation quality." Critical issue: invalid molecules get zero-vector ChemNet embeddings, creating a binary (valid/invalid) signal rather than the continuous quality gradient CLIP provides for images.

**3. Graph quantization destroys the uncertainty gradient (most fundamental)**
After diffusion, continuous outputs are discretized: adjacency → {0,1,2,3} bond types via argmax, node features → binary atom types. This creates a binary outcome per bond: either the perturbation is too small to flip the argmax (identical molecule), or it flips a bond (often breaking the molecule). No smooth middle ground exists. The Fisher bug (pre-fix) accidentally tested large perturbations (~230% std) — result: 99.5% of molecules got inf uncertainty. Current small perturbations (~3% std) often produce identical molecules after quantization. Both extremes fail; there is no useful regime.

**4. The last layer has limited rank**
ScoreNetworkA has only 93 last-layer parameters (0.15% of total), spanning only a 93-dimensional subspace of the 1444-dimensional output space ([38×38]). This limits the expressiveness of uncertainty — many output directions cannot be explored. However, even with more parameters, the quantization wall (point 3) would still prevent continuous uncertainty signals from surviving.

**5. FCD is unreliable at our sample sizes**
FCD estimates a 512x512 covariance matrix. At N=300 (top 30% of 1000), the matrix is rank-deficient (rank 299 < 512). The Frechet distance computation becomes numerically unstable. Additionally, filtering reduces diversity, which increases FCD even if individual molecules improve. The paper uses 10K samples.

### 6.2 What IS Working

Despite FCD not improving, the uncertainty does capture meaningful information:

1. **Validity prediction (nocorrect)**: Adj-only uncertainty is a strong predictor of molecular validity, especially at low step counts
2. **No inf/nan issues**: After the Fisher fix, all uncertainty values are finite
3. **Well-behaved distributions**: With correction, uncertainties follow a compact, well-separated distribution
4. **Network decomposition**: The separation into X-only and adj-only uncertainty reveals that adjacency uncertainty is more informative than node-feature uncertainty

### 6.3 Metric Considerations

**FCD limitations for this use case**:
- Distributional metric: filtering reduces diversity, which increases FCD
- Same encoder (ChemNet) used for both uncertainty and evaluation — circular dependency
- Requires >> 512 samples for reliable 512-dim covariance estimation
- Not directly comparable across different sample sizes

**Recommended alternatives**:
- **Validity rate**: Most direct measure; shows clear improvement with adj_only filtering
- **NSPDK MMD**: Already in GDSS codebase, kernel-based (no covariance estimation), more robust
- **Per-molecule properties**: QED, SA Score, LogP — robust to sample size
- **Scaffold diversity**: Verify filtering doesn't collapse to one "easy" scaffold

### 6.4 FCD: Not a Bug, a Limitation

FCD (Frechet ChemNet Distance) is the standard distributional metric for molecular generation, analogous to FID for images. However, it is **structurally inappropriate** for evaluating uncertainty-based filtering at our sample sizes. This is not a code bug — it is an inherent property of the metric's design.

**1. Covariance rank deficiency**: FCD computes the Frechet distance between two multivariate Gaussians fitted in ChemNet's 512-dimensional activation space. Fitting a 512x512 covariance matrix requires N >> 512 samples. When we filter to the top 30% of 1000 molecules (N=300), the covariance matrix has rank <= 299 < 512, making it rank-deficient. The matrix square root in the Frechet distance becomes numerically unstable, producing unreliable values.

**2. Diversity penalty**: FCD measures the distance between *distributions*, not individual molecule quality. Filtering removes molecules, which reduces the diversity (coverage) of the generated set. Even if every remaining molecule is individually better, the reduced coverage inflates FCD because the generated distribution no longer matches the test distribution's spread. This penalizes exactly the operation we're trying to evaluate.

**3. Reference set inconsistency (fixed)**: The original code subsampled the test set proportionally (`test_smiles[:len(valid_smiles)*2]`), meaning different filter fractions compared against different reference sets. This has been fixed to always use the full test set.

**Resolution**: We keep FCD in the evaluation for comparability with prior work, but add NSPDK MMD (kernel-based, no covariance estimation, already in the GDSS codebase) and per-molecule property metrics (mean QED, mean SA Score) as primary metrics. These do not suffer from the sample-size or diversity-penalty issues.

---

## 7. Open Questions & Next Steps

### Immediate (pending)
- [ ] Run low-step experiments (10, 20 steps) — Job: `eval_low_steps.job`
- [ ] Analyze whether uncertainty helps more at very low step counts
- [ ] Try NSPDK MMD as alternative to FCD

### Short-term improvements
- [ ] Extend Laplace to more layers (especially for ScoreNetworkA with only 93 last-layer params)
- [ ] Try Morgan fingerprints as alternative/complementary semantic encoder
- [ ] Tune prior precision (currently 1.0, may need hyperparameter search)
- [ ] Scale up to 10K molecules for reliable FCD comparison
- [ ] Report validity filtering curves as primary metric

### Medium-term directions
- [ ] Investigate pre-quantization uncertainty (measure score variance in continuous space)
- [ ] QM9 dataset experiments (checkpoints exist, no results yet)
- [ ] Explore whether uncertainty correlates with specific chemical properties
- [ ] Consider alternative uncertainty metrics (e.g., predictive variance in property space)

### Pivot: 3D Molecular Generation (proposed next direction)

The 2D graph domain has fundamental limitations for uncertainty-based filtering:
1. Chemical validity is binary (valid/invalid), and the Langevin corrector already enforces 100% validity
2. Graph quantization (continuous → discrete bond types) destroys the continuous uncertainty gradient
3. ChemNet only sees 2D SMILES — no geometric quality signal

**3D molecular diffusion models** (EDM, GeoLDM, EquiFM) are a better fit because:
- **Continuous quality spectrum**: No binary valid/invalid — instead strain energy, steric clashes, binding affinity, conformer quality
- **No corrector to eliminate the signal**: 3D geometry is too complex for a simple corrector to enforce
- **Richer semantic encoders**: 3D-aware networks (SchNet, DimeNet, Uni-Mol) capture geometric quality that 2D encoders miss
- **No quantization**: Atom coordinates remain continuous throughout — the uncertainty gradient is preserved
- **Practical value**: A molecule can be 2D-valid but 3D-terrible (strained, clashing, poor binding)

The Laplace + fixed-noise + semantic embedding pipeline transfers directly — swap the score network and encoder. The negative 2D result motivates why 3D is the right domain.

### For thesis
- [ ] Frame the negative FCD result as a domain gap finding (images → 2D molecular graphs)
- [ ] Position validity-based filtering as the key positive result for 2D
- [ ] Discuss implications for applying Bayesian UQ to discrete generative models
- [ ] Argue for 3D molecular generation as the natural next domain for this framework

---

## 8. Timeline

| Date | Milestone |
|------|-----------|
| Jan 19 - Feb 9 | Setup: Environment, data, checkpoints |
| Feb 9-10 | Core implementation: Laplace, ChemNet uncertainty, evaluation scripts |
| Feb 10-11 | First results: 10K molecules, step sweep, property analysis |
| Feb 11 | Network-level analysis (X vs Adj uncertainty) |
| Feb 17 | Fisher normalization bug fix |
| Feb 17-18 | Comprehensive evaluation: 5 step counts x 2 modes = 10 runs (Job 19584489) |
| Feb 19 | Deep analysis of results, identified 5 root causes for FCD degradation |
| Feb 19 | Planned: Low-step experiments (10, 20 steps) |
| TBD | QM9 experiments, metric improvements, thesis writing |

---

## Key Files

| File | Purpose |
|------|---------|
| `laplace_gdss_full.py` | Last-layer Laplace approximation for score networks |
| `chemnet_semantic_uncertainty.py` | ChemNet embeddings + entropy-based uncertainty |
| `evaluate_uncertainty_steps.py` | Main evaluation: generation + uncertainty + filtering |
| `evaluate_uncertainty_filtering.py` | Filtering-focused evaluation (Figure 3 reproduction) |
| `compare_comprehensive.py` | Cross-step comparison plots |
| `analyze_uncertainty.py` | Uncertainty vs molecular properties |
| `analyze_network_uncertainties.py` | X vs Adj network uncertainty analysis |
| `~/jobs/eval_comprehensive.job` | Job for 50-1000 step sweep |
| `~/jobs/eval_low_steps.job` | Job for 10-20 step experiments |
| `results_comprehensive/` | All results from comprehensive evaluation |
