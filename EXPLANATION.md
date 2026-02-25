# Generative Uncertainty for Molecular Graph Diffusion — Full Technical Explanation

**Author**: Pepijn Seij
**Date**: February 25, 2026

This document explains the full pipeline, what each component does, what the results mean, and what we learned.

---

## 1. What We're Doing (Big Picture)

We adapt [Jazbec et al. (2025) "Generative Uncertainty in Diffusion Models"](https://arxiv.org/abs/2502.01145) — originally designed for image generation — to **molecular graph generation** using the GDSS framework on the ZINC250k dataset.

**The core idea**: A trained diffusion model has a single set of "best" parameters (the MAP estimate). But there's a distribution of plausible parameters that also explain the training data. If we sample different plausible parameters and regenerate the same molecule (same random noise), and the results vary a lot, we're *uncertain* about that molecule. If results are consistent, we're *confident*.

**The hope**: High-uncertainty molecules are likely low-quality. By filtering them out, we can improve the quality of a generated molecular set.

---

## 2. The Diffusion Model: GDSS

GDSS (Graph Diffusion via the System of SDEs, [Jo et al., ICML 2022](https://arxiv.org/abs/2202.02514)) generates molecular graphs by running two coupled stochastic differential equations (SDEs) in reverse.

### 2.1 What It Generates

A molecule is represented as a 2D graph:
- **X** `[B, N, F]` — Node features (atom types). For ZINC250k: N=38 max atoms, F=9 atom types (C, N, O, F, P, S, Cl, Br, I) + 1 virtual "no atom" type
- **Adj** `[B, N, N]` — Adjacency matrix (bond types). Values represent: no bond, single, double, or triple bond

This is **2D molecular generation** — we get connectivity (which atoms, which bonds) but no 3D coordinates.

### 2.2 The Two Score Networks

GDSS uses two separate neural networks to learn the "score" (gradient of the log-probability) of the noisy data distribution:

#### ScoreNetworkX (Atom Types)
- **Purpose**: Predicts the noise added to node features at each diffusion timestep
- **Architecture**: 2-layer Graph Convolutional Network (DenseGCNConv) + 3-layer MLP
  - GCN layers: `DenseGCNConv(9→16)` → `tanh` → `DenseGCNConv(16→16)` → `tanh`
  - Features from all layers concatenated: `[9 + 16 + 16] = 41 dimensions`
  - Final MLP: `Linear(41→82)` → `elu` → `Linear(82→82)` → `elu` → `Linear(82→9)`
- **Total parameters**: 11,429
- **Last layer** (`final.linears.2`): `Linear(82→9)` = **747 parameters** (6.5% of total)
- **SDE type**: VP-SDE (Variance Preserving), `beta_min=0.1`, `beta_max=1.0`

#### ScoreNetworkA (Bond Types)
- **Purpose**: Predicts the noise added to the adjacency matrix at each diffusion timestep
- **Architecture**: 6 Attention layers + 3-layer MLP
  - Adjacency is expanded to multi-channel: `pow_tensor(adj, c_init=2)` gives `[B, 2, N, N]`
  - 6 `AttentionLayer` modules process both node features and adjacency channels
  - All adjacency channel outputs concatenated: `2 + 5×8 + 4 = 46 channels`
  - Final MLP: `Linear(46→92)` → `elu` → `Linear(92→92)` → `elu` → `Linear(92→1)`
- **Total parameters**: 62,873
- **Last layer** (`final.linears.2`): `Linear(92→1)` = **93 parameters** (0.15% of total)
- **SDE type**: VE-SDE (Variance Exploding), `sigma_min=0.2`, `sigma_max=1.0`

#### How They're Coupled
Both networks take **both** X and Adj as input. When ScoreNetworkX predicts the score for atom types, it conditions on the current bond structure, and vice versa. During sampling, updates alternate: correct X → correct Adj → predict X → predict Adj.

### 2.3 The Sampling Process (Predictor-Corrector)

Generation runs in reverse time, from pure noise (`t=1`) to clean data (`t≈0`), using N discretization steps (default: 1000).

At each timestep:
1. **Corrector** (Langevin dynamics): Refines the current sample using the score — takes a gradient step toward higher probability, with added noise for exploration. Uses adaptive step size based on signal-to-noise ratio (`snr=0.2`).
2. **Predictor** (Reverse Diffusion): Takes one step of the reverse SDE — the actual denoising step.

**Three sampling modes we test**:
- **Standard** (`correct`): Langevin corrector ON + valence correction ON — full GDSS pipeline
- **No valence correction** (`nocorrect`): Langevin corrector ON, post-hoc valence correction OFF
- **No corrector** (`nocorrector`): Langevin corrector replaced with `NoneCorrector` (does nothing), valence correction OFF — raw reverse diffusion only

### 2.4 Post-Processing: From Continuous to Discrete

The diffusion model outputs **continuous** tensors. To get actual molecules:
1. **Quantize adjacency**: `argmax` over 4 bond channels → discrete bond types {0,1,2,3}
2. **Binarize atoms**: threshold at 0.5 → discrete atom types
3. **Construct molecule**: Build RDKit `RWMol` atom-by-atom, bond-by-bond (`construct_mol`)
4. **Valence correction** (if enabled): Iteratively fix valence errors by downgrading bonds (`correct_mol`)
5. **Largest connected component**: Extract the largest fragment if the molecule is disconnected
6. **SMILES conversion**: Canonicalize via RDKit

This quantization step is important — it means small parameter perturbations can produce **identical** discrete molecules, destroying the continuous uncertainty signal.

---

## 3. The Uncertainty Method

### 3.1 Last-Layer Laplace Approximation

We approximate the Bayesian posterior over the **last linear layer** of each score network.

**Why last layer only?** Full-network Laplace is computationally prohibitive. Last-layer Laplace is standard practice (Daxberger et al., 2021) and is what Jazbec et al. use. The layer targeted is `final.linears.2` — the output layer of the 3-layer MLP that produces the final score prediction.

**The Fisher Information Matrix** (diagonal approximation):
1. For each training batch, compute the score matching loss: `L = mean((predicted_score - target_score)²)`
2. Backpropagate to get gradients of the last-layer parameters
3. Accumulate **squared gradients**: `H += grad²` (this is the diagonal empirical Fisher)
4. The Fisher is a **sum** over all training samples (not an average — this was a bug we fixed)

**The posterior**:
```
q(θ|D) = N(θ_MAP, Σ)
where Σ = diag(1 / (H + α))
```
- `θ_MAP` = trained weights (maximum a posteriori)
- `H` = diagonal Fisher information (sum of squared gradients)
- `α` = prior precision (1.0) — regularization toward zero

**Posterior standard deviations** (after Fisher fix):
- X-network: ~0.029 (parameters perturbed by ~3% from MAP)
- Adj-network: ~0.017 (parameters perturbed by ~2% from MAP)

### 3.2 The Fisher Normalization Bug (Fixed Feb 17)

The original code divided the Fisher by `n_data` (number of training batches), turning the sum into an average. This made the posterior ~30x too diffuse for the X-network:
- **Before fix**: X posterior std ≈ 0.95 (parameters randomly scrambled — 230% perturbation)
- **After fix**: X posterior std ≈ 0.029 (reasonable 3% perturbation)

With the bug, 99.5% of molecules got `unc_x = inf` because the wildly perturbed X-network produced garbage. Only Adj-only results were valid in pre-fix runs.

### 3.3 Fixed-Noise Uncertainty Estimation

Following Jazbec et al. Algorithm 1:

1. **Fix the random noise** z = (z_x, z_adj) — every posterior sample starts from the same noise
2. **Generate with MAP parameters**: molecule_0 = generate(θ_MAP, z) → get ChemNet embedding e_0
3. **For m = 1..M (M=10 posterior samples)**:
   - Sample θ_m ~ q(θ|D)
   - Generate molecule_m = generate(θ_m, z) with the **same noise z**
   - Get ChemNet embedding e_m
4. **Compute uncertainty** = entropy of the distribution over {e_0, e_1, ..., e_M}

By fixing the noise, we isolate **epistemic uncertainty** (what the model doesn't know) from **aleatoric uncertainty** (inherent randomness).

### 3.4 Three Uncertainty Decompositions

We go beyond the original paper by decomposing uncertainty by network:

- **Combined**: Both θ_x and θ_adj sampled simultaneously — total epistemic uncertainty
- **X-only**: Only θ_x sampled, θ_adj fixed at MAP — uncertainty from atom type prediction alone
- **Adj-only**: Only θ_adj sampled, θ_x fixed at MAP — uncertainty from bond prediction alone

This tells us **which network** contributes more to uncertainty. Finding: Adj-only is consistently more informative for predicting molecular quality.

### 3.5 ChemNet Semantic Embeddings

We use ChemNet (the same network underlying the FCD metric) as our semantic encoder:
- **Architecture**: Neural network trained on ~6,000 biological assays
- **Output**: 512-dimensional embedding from the penultimate layer
- **Input**: SMILES strings (2D molecular structure)
- **Invalid molecules**: Get **zero vectors** (all 512 dims = 0). These naturally increase variance when mixed with valid embeddings, correctly producing high uncertainty.

### 3.6 The Entropy Formula

For each molecule, given M+1 embeddings (MAP + M posterior samples) of dimension D=512:

```
H = 0.5 × D × (1 + ln(2π)) + 0.5 × Σ_d ln(var_d + σ²)
```

Where:
- `var_d` = variance of the d-th embedding dimension across the M+1 samples
- `σ² = 0.001` = observation noise (prevents log(0), smoothing floor)
- H is the differential entropy of a diagonal Gaussian fitted to the embeddings

**Interpretation**:
- **Low entropy** (e.g., -500): All posterior samples produce nearly identical molecules → high confidence
- **High entropy** (e.g., +1905 sentinel): All posterior samples produce completely different/invalid molecules → maximum uncertainty
- With correction (100% valid): range is approximately [-620, -300], compact and unimodal
- Without correction (mixed valid/invalid): bimodal — valid molecules cluster at ~-500, invalid ones spike at ~+1905

---

## 4. Results and What They Mean

### 4.1 The Central Negative Result: Filtering Hurts FCD With Correction

When the Langevin corrector is active, **validity is 100% at all step counts** (50-1000). There are no bad molecules to filter. Uncertainty filtering consistently worsens FCD by 1.5-3.5 points compared to random selection.

**Why?** The uncertainty metric captures how "unusual" a molecule is — how much the model's output varies with parameter perturbation. But unusual ≠ bad. The corrector ensures all molecules are chemically valid, so the most "uncertain" molecules are simply the most *novel* ones — and removing novelty hurts distributional diversity, which inflates FCD.

### 4.2 The Positive Result: Validity Prediction Without Correction

At **50 steps without correction** (53.3% validity), adj-only uncertainty successfully separates valid from invalid molecules:

- The scatter plot (X vs Adj uncertainty, colored by validity) shows three clear clusters:
  - **Bottom-left** (low X, low Adj): Valid molecules
  - **Top band** (Adj ≈ 1905): Molecules where bond prediction completely failed
  - **Top-right** (both high): Both networks failed
- Adj-only filtering at 40% keeps → FCD improves by 1.2 points vs random
- Validity of kept molecules: 85% (adj-only) vs 55% (random) at 50% filtering

This works because adj-only uncertainty directly captures **bond structure failure**, which is the primary mode of invalidity.

### 4.3 Uncertainty Distribution Shapes

**With correction** (all valid):
- All three types (combined, x-only, adj-only) show compact, roughly Gaussian distributions
- Range: approximately [-620, -300] (entropy units)
- No inf values, no sentinel values
- X and Adj uncertainties are **weakly correlated** (r ≈ 0.15), meaning they capture different aspects
- Combined correlates more with X (r ≈ 0.26) than Adj (r ≈ 0.17)

**Without correction** (mixed valid/invalid):
- **Bimodal**: Valid molecules at ~-500, invalid molecules spiking at ~+1905 (sentinel)
- The sentinel value comes from the all-invalid penalty: `0.5 × 512 × (1 + ln(2π)) + 0.5 × 512 × ln(100)` ≈ 1905.4
- At 50 steps: ~21.6% of combined, ~24.5% of x-only, ~39.4% of adj-only values are sentinels
- Adj produces the most sentinels because bond structure is harder to get right without correction
- Correlations become stronger in nocorrect mode (r up to 0.63-0.72) because shared extreme values from invalid molecules dominate

### 4.4 Which Uncertainty Type Is Best?

**Adj-only wins almost everywhere.** Average FCD improvement over random:

| Mode | Combined | X-only | Adj-only |
|------|----------|--------|----------|
| Correct (avg across steps) | -1.63 | -1.65 | **-1.39** |
| Nocorrect, 50 steps | -1.59 | -0.75 | **+0.60** |
| Nocorrect, 100 steps | -1.00 | -0.99 | **-0.80** |

Adj-only has the smallest penalty in correct mode and is the **only type that achieves positive improvement** in the nocorrect/low-step regime.

**Why Adj dominates**: The adjacency matrix defines the molecular scaffold — which atoms are connected by which bonds. This is the primary determinant of chemical validity. The X network assigns atom types to existing positions, which is a simpler task. When things go wrong, it's usually the bond structure that fails.

### 4.5 Step Count Effects

**FCD baseline** (full set, no filtering):
| Steps | Correct | Nocorrect |
|-------|---------|-----------|
| 50 | 21.5 | 20.6 |
| 100 | **16.5** | **16.2** |
| 200 | 16.8 | 16.9 |
| 500 | 16.6 | 16.6 |
| 1000 | 16.9 | 17.1 |

- **100 steps is optimal** — best FCD in both modes
- 50 steps significantly degrades quality (FCD ~21 vs ~16-17)
- 500-1000 steps offer no improvement over 100, possibly slight degradation from accumulated numerical error
- Validity without correction improves with more steps: 53% (50) → 86% (100) → 96% (1000)

### 4.6 Property Correlations (from the 10k run)

From 10,000 molecules at 1000 steps with correction:
- **Molecular weight**: Spearman ρ = 0.21 (modest positive correlation — larger molecules are more uncertain)
- **QED** (drug-likeness): ρ = 0.07 (near zero)
- **SA Score** (synthetic accessibility): ρ = 0.01 (near zero)
- **LogP** (hydrophobicity): ρ = 0.00 (zero)

**Interpretation**: Uncertainty primarily reflects molecular **size** (more atoms = more parameters to get right = more room for variation). It does **not** capture chemical quality (QED, SA, LogP). This further explains why filtering by uncertainty doesn't improve the set — it's removing large molecules, not bad ones.

---

## 5. Why It Doesn't Work Like the Paper

The paper demonstrates clear FCD/FID improvements on ImageNet. Our molecular application fails. Five root causes:

### 5.1 The Langevin Corrector Eliminates the Signal
The Langevin corrector runs at every diffusion step, iteratively pushing the sample toward higher probability. It effectively "fixes" molecules that would otherwise be invalid. This correction mechanism doesn't exist in standard image diffusion — images don't have a binary valid/invalid criterion.

### 5.2 ChemNet Is Not CLIP
CLIP learns rich perceptual quality from 400M image-text pairs. A blurry image and a sharp image get very different CLIP embeddings. ChemNet, trained on ~6K bioassays, gives a **binary signal**: valid molecules get meaningful 512-dim embeddings, invalid molecules get zero vectors. There's no continuous "quality gradient" — a slightly bad molecule and a very bad molecule both map to zero.

### 5.3 The Quantization Wall (Most Fundamental Issue)
Images are continuous — small parameter perturbations produce slightly different pixel values, giving a smooth uncertainty gradient. Molecular graphs are **discrete** — continuous adjacency values are quantized to {no bond, single, double, triple} via `argmax`. This creates a binary outcome for each bond: either the perturbation is too small to flip the argmax (producing an identical molecule), or it flips a bond type (often breaking the molecule entirely). There is no smooth middle ground.

**We accidentally tested this**: The Fisher normalization bug (pre-fix) gave the X-network a posterior std of ~0.95 (~230% perturbation) — effectively "order of magnitude larger" perturbations like in the Jazbec paper. The result was **catastrophic**: 99.5% of molecules got `unc_x = inf` because the perturbed parameters produced garbage. After the fix, perturbations are ~3%, which is small enough that most molecules don't change after quantization.

So we've seen both extremes — too small (no signal) and too large (all broken) — with no useful regime in between. This is a fundamental limitation of applying continuous-space uncertainty to discrete outputs.

### 5.4 The Last Layer Has Limited Rank
ScoreNetworkA has only **93 last-layer parameters** (0.15% of total). These 93 parameters can only span a 93-dimensional subspace of perturbations in the 1444-dimensional output space (`[38×38]`). This limits the *expressiveness* of the uncertainty — many directions in output space simply cannot be explored. However, even with more parameters, the quantization wall (5.3) would still prevent continuous uncertainty signals from surviving.

### 5.5 FCD Is Unreliable at N=1000
FCD estimates a 512×512 covariance matrix. At N=300 (top 30% of 1000), the covariance is rank-deficient (rank 299 < 512). The Frechet distance becomes numerically unstable. Additionally, filtering reduces diversity, which inflates FCD even if individual molecules improve. The paper uses 10K samples.

---

## 6. What IS Working

1. **Validity prediction**: Adj-only uncertainty reliably identifies invalid molecules when the corrector is disabled — up to +21.6 percentage points improvement in validity at 50 steps
2. **Network decomposition**: Separating X vs Adj uncertainty reveals that bond structure is the dominant failure mode — a novel insight not in the original paper
3. **Well-behaved posteriors**: After the Fisher fix, all values are finite with interpretable distributions
4. **The framework is sound**: The Laplace + fixed-noise + semantic embedding pipeline works correctly; the limitation is domain-specific

---

## 7. Proposed Pivot: 3D Molecular Generation

The negative result on 2D graphs motivates a move to 3D molecular generation.

### 7.1 Why 3D Solves All Five Root Causes

| Limitation in 2D | How 3D Solves It |
|-------------------|-----------------|
| **Quantization wall** (§5.3) | Atom coordinates are continuous — no argmax discretization, uncertainty propagates naturally |
| **Binary valid/invalid** (§5.1) | Quality is a continuous spectrum: strain energy, bond length deviations, clash scores |
| **Corrector enforces 100% validity** (§5.1) | 3D models have no Langevin corrector or valence correction — raw generation quality varies |
| **ChemNet sees only 2D SMILES** (§5.2) | 3D-aware encoders (Uni-Mol, SchNet, PaiNN) produce high-dimensional embeddings from coordinates |
| **No practical quality signal** (§5.4) | Real applications: docking scores, conformer quality, relaxation energy |

### 7.2 All Candidate Models & Papers

| Model | GitHub | Paper |
|-------|--------|-------|
| **EDM** | https://github.com/ehoogeboom/e3_diffusion_for_molecules | https://arxiv.org/abs/2203.17003 |
| **GeoLDM** | https://github.com/MinkaiXu/GeoLDM | https://arxiv.org/abs/2305.01140 |
| **EQGAT-diff** | https://github.com/jule-c/eqgat_diff | https://arxiv.org/abs/2309.17296 |
| **MiDi** | https://github.com/cvignac/MiDi | https://arxiv.org/abs/2302.09048 |
| **GCDM** | https://github.com/BioinfoMachineLearning/bio-diffusion | https://arxiv.org/abs/2302.04313 |
| **EquiFM/MolFM** | https://github.com/AlgoMole/MolFM | https://arxiv.org/abs/2312.07168 |
| **SemlaFlow** | https://github.com/rssrwn/semla-flow | https://arxiv.org/abs/2408.13155 |
| **Uni-Mol** (encoder) | https://github.com/deepmodeling/Uni-Mol | https://arxiv.org/abs/2303.16982 |

### 7.3 Model Deep Dives

#### EDM (Hoogeboom et al., ICML 2022) — PRIMARY RECOMMENDATION

- **Architecture**: EGNN (E(n) Equivariant Graph Neural Network) + DDPM-style diffusion, epsilon-parameterization
- **Repo**: 549 stars, 139 forks, MIT license — largest community of all candidates
- **Score network**: 256 hidden features, 9 EGNN layers (QM9), 4 layers (GEOM-Drugs)
- **Last layer for Laplace**: `self.embedding_out = nn.Linear(hidden_nf, out_node_nf)` — a standard `nn.Linear`, directly analogous to GDSS's `final.linears.2`
- **Dual output stream**: Feature predictions go through the clean `embedding_out` layer; coordinate predictions are velocity residuals `vel = x_final - x` with no final linear layer (coordinates are updated iteratively through `EquivariantUpdate` blocks with their own `coord_mlp`)
- **Pretrained checkpoints**: Available for QM9 (conditional on 6 properties) and GEOM-Drugs
- **Equivariance**: E(3) — includes reflections, so chirality-blind

**QM9 metrics** (independent benchmark, ACS Omega 2025):

| Metric | EDM |
|--------|-----|
| Atom Stability | 99.20% |
| Molecule Stability | 89.59% |
| Validity | 91.59% |
| Uniqueness | 99.34% |
| Novelty | 93.76% |
| Connectivity | 99.34% |

**GEOM-Drugs**: 97.6% validity but only **37% connectivity** (worst of all models — the main weakness).

**Laplace feasibility: HIGH.** The `embedding_out = nn.Linear(256, out)` layer is identical in structure to what we already implemented in GDSS. The `GDSSLaplaceFull` pattern transfers almost 1:1: freeze all but the last linear layer, compute diagonal empirical Fisher on the noise prediction MSE loss, sample from the posterior.

**Training cost**: ~7 days on 1x GTX 1080Ti (QM9, 1100 epochs). Can skip training by using pretrained checkpoints.

#### GeoLDM (Xu et al., ICML 2023)

- **Architecture**: Two-stage model — (1) E(n)-equivariant VAE encodes molecules into a latent space, (2) EGNN-based diffusion runs in that latent space
- **Built directly on the EDM codebase** — same directory structure, same EGNN implementation, same file names
- **Consistently better than EDM**: +6% molecule stability on QM9, +7% validity on GEOM-Drugs
- **Pretrained**: Yes (Google Drive, QM9 + GEOM-Drugs)
- **Code**: 271 stars, MIT license, but stale since Feb 2023 (12 open issues)
- **Key innovation**: "Point-structured latent space" preserving both equivariant (coordinate) and invariant (feature) components

**Laplace issue**: The diffusion operates in **latent space**, not data space. Uncertainty on latent-space score predictions would need to be propagated through the decoder to get meaningful molecular uncertainty. This adds complexity and potential information loss. Also, there are THREE separate EGNNs (encoder, decoder, diffusion dynamics) — which one gets Laplace?

**Verdict**: Skip for now. Since it uses the exact same EGNN architecture as EDM, any Laplace implementation built for EDM transfers directly to GeoLDM later if needed.

#### EQGAT-diff (Le et al., ICLR 2024)

- **Architecture**: E(3)-equivariant graph attention network with denoising diffusion, **x0-parameterization** (predicts clean data, not noise)
- **Three output modalities**: continuous coordinates, categorical atom types (softmax), categorical bond types (softmax)
- **Adaptive noise scheduling**: Time-dependent loss weighting `w(t) = clamp(SNR(t), 0.05, 1.5)` focusing on the critical transition regime
- **Pretrained**: Yes (Google Drive) — QM9, GEOM-Drugs (fine-tuned from PubChem3D pretraining on 4x A100 for 24h)
- **Code**: 5 stars, MIT license, minimal documentation ("this repository serves as a placeholder")

**Performance**: Best GEOM-Drugs FCD (5.1, tied with SemlaFlow), 88% connectivity (vs EDM's 37%).

**Laplace feasibility: MODERATE.** The multi-head output requires mixed Fisher computation: Gaussian likelihood for coordinates, categorical cross-entropy for atom/bond types. This requires modifying the Fisher computation from GDSS, but is not fundamentally harder.

**Verdict**: Strong second choice if EDM's poor GEOM-Drugs connectivity is a problem.

#### MiDi (Vignac et al., ECML-PKDD 2023)

- **Architecture**: Graph Transformer with **mixed diffusion** — Gaussian noise for 3D coordinates, multinomial noise for atom types and bond types
- **Joint bond generation**: Bonds are generated as part of the diffusion process, not determined post-hoc from coordinates. This means uncertainty in the score network **directly propagates to bond predictions** — unlike EDM/GCDM where bonds are inferred by a separate rule-based step.
- **Three explicit output MLPs** ending in `nn.Linear`:
  - `mlp_out_X`: node features → atom types + charges
  - `mlp_out_E`: edge features → bond types
  - `mlp_out_pos`: PositionsMLP → coordinate updates
- **Code**: Clean, well-organized, Hydra configs. Based on DiGress codebase.
- **Pretrained**: Partial — some checkpoints lost due to Google Drive account deletion (July 2024). QM9 explicit H (adaptive) and GEOM implicit H available.

**Known bug**: Aromatic bond valency counted as 1 instead of 1.5, inflating stability metrics. Independent benchmark found 84.4% mol stability on QM9 (vs 97.5% self-reported).

**Laplace feasibility: GOOD — cleanest output structure of all models.** Each output head is an `nn.Sequential` ending in a standard `nn.Linear` layer. The transformer backbone (feature extraction) is cleanly separated from prediction heads — exactly the architecture Jazbec et al. target.

**Why MiDi is interesting for uncertainty**: The mixed discrete/continuous diffusion avoids our "quantization wall" from 2D. Bond types are probabilistic throughout the generation process, not argmax-discretized at the end. This is directly relevant to our GDSS finding.

**Verdict**: Best model for studying bond-level uncertainty propagation. Some checkpoints missing.

#### GCDM (Morehead & Cheng, Nature Communications Chemistry 2024)

- **Architecture**: GCPNet++ (Geometry-Complete Perceptron Network) — maintains separate scalar (invariant) and vector (equivariant) features at both node and edge levels. 9 GCP message-passing layers.
- **SE(3) equivariant** (rotations + translations, NOT reflections) — the ideal symmetry group for molecules since chirality matters. EDM is E(3) which is chirality-blind.
- **Best NLL** on QM9 (-171.0 vs EDM's -110.7), best PoseBusters validity on GEOM-Drugs (77% vs GeoLDM's 38%)
- **Pretrained**: Yes (Zenodo, ~5GB) — QM9 unconditional + conditional, GEOM-Drugs unconditional
- **Code**: Well-structured, PyTorch Lightning + Hydra

**Laplace feasibility: MODERATE-HARD.** The output goes through `scalar_node_projection_gcp`, a nested GCP module with internal linear layers that mixes scalar and vector channels before projecting to scalars. Not a simple `nn.Linear` — you would need to surgically extract the innermost linear layer from the GCP module.

**Verdict**: Best pure generative quality but hardest Laplace adaptation. Skip unless EDM proves insufficient.

#### EquiFM / MolFM (Song et al., NeurIPS 2023)

- **Architecture**: Conditional Flow Matching with equivariant optimal transport for coordinates, information-aligned paths for categorical features
- **Code**: 41 stars, Docker-only, 4 commits total, no training scripts, no checkpoints — essentially unusable
- **Verdict**: Skip entirely. For flow matching, use SemlaFlow instead.

#### SemlaFlow (Irwin et al., 2024)

- **Architecture**: SE(3)-equivariant transformer + flow matching — only **20 sampling steps** needed (vs 1000 for EDM)
- **Code**: 54 stars, MIT license, clean
- **Pretrained**: Yes (QM9 + GEOM-Drugs)
- **Jazbec et al. explicitly covers flow matching** in Appendix C.7: "We demonstrate the applicability of our framework beyond diffusion models by applying it to a (latent) flow matching model"
- **Flow matching advantage**: Deterministic ODE integration means no stochastic corrector step. Uncertainty is injected purely through weight perturbation. Fewer steps = less compute per posterior sample.

**Verdict**: Best option if we want to explore beyond DDPM-style diffusion. The 20-step sampling makes posterior sampling much cheaper.

### 7.4 Model Comparison Summary

| Model | Laplace Ease | QM9 Quality | GEOM-Drugs | Code Quality | Pretrained | Recommendation |
|-------|-------------|-------------|------------|-------------|------------|----------------|
| **EDM** | HIGH | Baseline | Poor connectivity (37%) | Excellent (549★) | Yes | **Primary** |
| GeoLDM | Same as EDM | Better (+6%) | Better (+7%) | Good (271★, stale) | Yes | Easy migration later |
| EQGAT-diff | Moderate | Good | Best FCD (5.1) | Fair (5★, sparse) | Yes | Strong second |
| **MiDi** | GOOD | Good 3D geometry | Good stability | Clean (Hydra) | Partial | Best for bond uncertainty |
| GCDM | Hard | Best NLL | Best PB-validity | Good (PL+Hydra) | Yes (Zenodo) | Best quality, hardest |
| EquiFM | N/A | Good | N/A | Unusable | No | Skip |
| SemlaFlow | Moderate | SOTA | Good | Clean (MIT) | Yes | Flow matching option |

### 7.5 Semantic Encoder: Uni-Mol (Clear Winner)

For computing semantic uncertainty in 3D, we need a pretrained 3D-aware molecular encoder. **Uni-Mol** (Zhou et al., ICLR 2023) is the clear best choice.

**Why Uni-Mol**:
- **Scale**: Pretrained on **209M molecular 3D conformations** (vs SchNet/PaiNN on 134k QM9 — 1500x more data)
- **Architecture**: SE(3)-invariant Transformer with spatial positional encoding, 15 layers, 512 embedding dim
- **[CLS] token**: Built-in molecule-level 512-dim embedding — no manual atom pooling needed
- **Truly 3D-aware**: Takes atom types AND 3D coordinates as input, uses interatomic distances in attention
- **Custom coordinate input**: Accepts raw `{'atoms': [...], 'coordinates': [...]}` dicts — we can feed diffusion model output directly without converting to SMILES

**Installation and usage**:
```python
pip install unimol_tools  # auto-downloads pretrained weights from HuggingFace

from unimol_tools import UniMolRepr
clf = UniMolRepr(data_type='molecule', remove_hs=False)

# From raw 3D coordinates (our use case):
data = {
    'atoms': [['C', 'C', 'O'], ['C', 'C', 'C', 'C', 'C', 'C']],
    'coordinates': [coords_array_1, coords_array_2]  # numpy (n_atoms, 3)
}
reprs = clf.get_repr(data, return_atomic_reprs=True)
cls_embeddings = reprs['cls_repr']    # shape: (n_mols, 512)
```

**Key advantage for semantic uncertainty**: SE(3)-invariance means different posterior samples that produce the same molecule in different orientations get identical embeddings. Different 3D conformers of the same molecule get different embeddings — this lets us detect geometric uncertainty specifically.

**Alternatives considered and rejected**:
| Encoder | Pretraining | Embedding | Verdict |
|---------|------------|-----------|---------|
| **Uni-Mol** | 209M conformations | 512-dim [CLS] | **Use this** |
| COATI | Large (proprietary+public) | 256-dim | Fallback option |
| SchNet | QM9 (134k) | 30-128 | Too small pretraining |
| PaiNN | QM9 (134k) | ~128 | Too small pretraining |
| GemNet | OC20 (catalysis) | 512 | Wrong domain |
| Frad | QM9/MD17 | Varies | No clean API |

### 7.6 Quality Metrics for 3D

The key advantage of 3D: **continuous quality metrics** replace binary valid/invalid.

#### Tier 1 — Primary Metrics (Must Have)

**1. GFN2-xTB Strain Energy** — the gold standard
- `E_strain = E_generated - E_relaxed` using semi-empirical quantum chemistry
- High strain = geometry far from local minimum = low quality
- ~1-10 seconds/molecule (parallelizable), accurate to ~2 kcal/mol
- Used by GEOM dataset for energy annotation
- Install: `conda install -c conda-forge xtb-python`

```python
from ase import Atoms
from ase.optimize import BFGS
from xtb.ase.calculator import XTB

atoms = Atoms(numbers=atomic_numbers, positions=positions)
atoms.calc = XTB(method="GFN2-xTB")
E_gen = atoms.get_potential_energy()
# ... optimize geometry ...
E_strain = E_gen - E_relaxed  # always >= 0
```

**2. MMFF94 Strain Energy** — fast fallback
- Same concept but using classical force field (RDKit, no extra deps)
- ~milliseconds/molecule
- Less accurate but useful for screening

**3. Bond Length W1** — distribution-level geometric quality
- Wasserstein-1 distance between generated and reference bond length distributions
- Introduced by MiDi as standard 3D quality metric

#### Tier 2 — Supporting Metrics (Should Have)

4. **Bond Angle W1 + Dihedral W1** — capture angular quality beyond distances
5. **Clash Score** — fraction of atom pairs with steric clashes (continuous, 0-1)
6. **Validity + Atom/Molecule Stability** — for comparability with published results

#### Tier 3 — Nice to Have

7. FCD, QED/SA distributions, PoseBusters pass rate, TFD (Torsion Fingerprint Deviation)

**PoseBusters** (`pip install posebusters`) provides a ready-made suite of geometric validity checks: bond lengths, bond angles, planar rings, steric clashes, internal energy.

### 7.7 Datasets

**Phase 1: QM9** (development)
- 130,831 molecules, max 9 heavy atoms (up to 29 with H), 5 elements (C, H, N, O, F)
- DFT-optimized geometries (B3LYP/6-31G(2df,p))
- Standard split: 100k / 18k / 13k (train/val/test)
- EDM auto-downloads QM9 when running `main_qm9.py`
- **Why start here**: Fast iteration (hours to train EDM), well-understood baselines, small molecules = fast uncertainty computation

**Phase 2: GEOM-Drugs** (scaling)
- 304,466 drug-like molecules, avg 44 atoms (24.9 heavy), up to 181 atoms
- ~102 conformers per molecule (generated with CREST/GFN2-xTB)
- Download: Harvard Dataverse `rdkit_folder.tar.gz` (~50GB)
- EDM's `build_geom_dataset.py` selects 30 lowest-energy conformers per molecule, 80/10/10 split
- **Why scale here**: Harder benchmark, drug-relevant molecules, more room for uncertainty filtering to improve results
- **Caution**: GEOM-Drugs Revisited (2025) found bugs in valency computation and bond order calculation that inflated published stability metrics

### 7.8 Implementation Plan

#### Phase 1: EDM + QM9 (~2 weeks)
1. Clone EDM repo, load pretrained QM9 checkpoint
2. Adapt `GDSSLaplaceFull` → `EDMLaplace`: target `embedding_out` layer in EGNN
3. Compute diagonal Fisher on QM9 training data (noise prediction MSE loss)
4. Generate 10k molecules with posterior sampling (fixed noise, M=10 posterior samples)
5. Compute GFN2-xTB strain energy + geometric metrics for each molecule
6. Plot uncertainty filtering curves: x = fraction kept, y = mean strain energy (should decrease)

#### Phase 2: Semantic Uncertainty (~1 week)
7. Install Uni-Mol in `gen_uncertainty` env
8. Write `unimol_semantic_uncertainty.py` (analogous to `chemnet_semantic_uncertainty.py`)
9. Encode posterior samples with Uni-Mol, compute eigenvalue-based semantic entropy
10. Compare epistemic (Laplace) vs semantic (Uni-Mol) uncertainty for filtering quality

#### Phase 3: Scale + Extend (~2-3 weeks)
11. Scale to GEOM-Drugs (use pretrained checkpoint or retrain)
12. Optionally try MiDi for bond-level uncertainty propagation
13. Consider SemlaFlow for flow matching comparison (Jazbec Appendix C.7)

**Total estimate: ~4-6 weeks for full 3D uncertainty framework**

---

## 8. Key Code Files

| File | What It Does |
|------|-------------|
| `laplace_gdss_full.py` | Diagonal Laplace approximation: Fisher fitting, posterior sampling, parameter perturbation |
| `chemnet_semantic_uncertainty.py` | ChemNet embeddings, entropy computation, the full Jazbec algorithm |
| `evaluate_uncertainty_steps.py` | Main evaluation: generation with variable steps, 3 uncertainty types, filtering metrics |
| `evaluate_uncertainty_filtering.py` | Simpler evaluation (Figure 3 style), combined uncertainty only |
| `solver.py` | PC sampler, Langevin corrector, NoneCorrector, reverse SDE integration |
| `sde.py` | VPSDE and VESDE definitions, forward/reverse processes |
| `models/ScoreNetwork_X.py` | GCN + MLP for atom type scores |
| `models/ScoreNetwork_A.py` | Attention + MLP for bond type scores |
| `utils/mol_utils.py` | Molecule construction, valence checking, SMILES conversion |
| `compare_comprehensive.py` | Cross-step/cross-mode comparison plots |
| `analyze_uncertainty.py` | Property correlations, per-decile analysis |
| `analyze_network_uncertainties.py` | X vs Adj decomposition analysis |

---

## 9. Glossary

| Term | Meaning |
|------|---------|
| **MAP** | Maximum A Posteriori — the trained model weights (best single point estimate) |
| **Posterior** | Distribution over plausible model parameters given the training data |
| **Fisher information** | Curvature of the loss landscape — measures how much the loss changes when parameters change |
| **Laplace approximation** | Approximate the posterior as a Gaussian centered at the MAP, with covariance from the inverse Hessian (here: inverse Fisher) |
| **Epistemic uncertainty** | Uncertainty due to limited training data — reducible with more data |
| **Aleatoric uncertainty** | Inherent randomness in the generation process — irreducible |
| **FCD** | Frechet ChemNet Distance — distributional distance between generated and reference molecules in ChemNet embedding space (lower = better) |
| **NSPDK MMD** | Neighborhood Subgraph Pairwise Distance Kernel Maximum Mean Discrepancy — kernel-based graph similarity metric |
| **QED** | Quantitative Estimate of Drug-likeness — scalar [0,1] combining molecular properties (higher = more drug-like) |
| **SA Score** | Synthetic Accessibility Score — how easy a molecule is to synthesize [1-10] (lower = easier) |
| **VP-SDE** | Variance Preserving SDE — noise schedule where signal+noise variance stays constant (used for atom types) |
| **VE-SDE** | Variance Exploding SDE — noise schedule where variance grows monotonically (used for bonds) |
| **PC Sampler** | Predictor-Corrector sampler — alternates between reverse-SDE steps (predictor) and Langevin refinement (corrector) |
| **Sentinel value** | The maximum uncertainty value (~1905.4) assigned when all M posterior samples produce invalid molecules |
