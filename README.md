# Generative Uncertainty for Molecular Generation (GDSS)

Implementation of **Jazbec et al. (2025) "Generative Uncertainty in Diffusion Models"** for molecular generation using the GDSS framework.

## Overview

This implementation adds uncertainty quantification to GDSS molecular generation using:
- **Last-Layer Laplace Approximation** for posterior estimation
- **ChemNet embeddings** as semantic likelihood (512-dim learned representations)
- **Posterior predictive uncertainty** following Algorithm 1 from Jazbec et al.

The key insight: For a fixed noise z, sample M parameter sets from the posterior, generate M molecules, and measure variability in ChemNet embedding space.

---

## Codebase Structure

### Core GDSS Files (Original)

| File | Description |
|------|-------------|
| `models/ScoreNetwork_X.py` | Score network for node features (atom types) |
| `models/ScoreNetwork_A.py` | Score network for adjacency matrix (bonds) |
| `sde.py` | VP-SDE and VE-SDE implementations |
| `sampler.py` | Original GDSS sampling methods |
| `trainer.py` | Training loop for score networks |
| `main.py` | Original training/sampling entry point |

### Uncertainty Estimation Files (New)

| File | Description |
|------|-------------|
| `laplace_gdss_full.py` | **Core Laplace implementation** - diagonal Fisher approximation for score networks |
| `chemnet_semantic_uncertainty.py` | **ChemNet semantic encoder** - converts molecules to 512-dim embeddings |
| `evaluate_uncertainty_filtering.py` | **Main evaluation script** - reproduces Figure 3 from Jazbec et al. |
| `run_mol_semantic_uncertainty.py` | Simple pipeline for testing uncertainty computation |

### Data & Utils

| File | Description |
|------|-------------|
| `data/preprocess.py` | Preprocess molecular datasets (ZINC250k, QM9) |
| `utils/mol_utils.py` | Molecule conversion utilities (tensors → RDKit → SMILES) |
| `utils/graph_utils.py` | Graph manipulation utilities |
| `utils/loader.py` | Data loading utilities |

---

## Detailed File Descriptions

### 1. `laplace_gdss_full.py` - Laplace Approximation

**Purpose**: Implements Last-Layer Laplace Approximation for GDSS score networks.

**Key Classes**:
- `GDSSLaplaceFull`: Wraps a score network with Laplace approximation
  - `fit()`: Computes diagonal empirical Fisher information
  - `sample_parameters()`: Samples from Gaussian posterior q(θ|D)
  - `get_posterior_std()`: Returns uncertainty over parameters

**Key Functions**:
- `create_laplace_models()`: Creates Laplace wrappers for both score networks
- `fit_laplace_models()`: Fits Fisher approximation on training data
- `uncertainty_aware_sampling()`: Generates samples with optional fixed noise (critical for Jazbec-style uncertainty)

**Algorithm (Fisher fitting)**:
```
For each training batch:
    1. Sample random timestep t
    2. Add noise to clean graph: x_t, adj_t = SDE.perturb(x_0, adj_0, t)
    3. Forward pass: score = model(x_t, adj_t, flags)
    4. Compute loss: ||score - target||²
    5. Backward pass to get gradients
    6. Accumulate squared gradients: H += grad²
```

### 2. `chemnet_semantic_uncertainty.py` - Semantic Encoder

**Purpose**: Extracts ChemNet embeddings and computes semantic uncertainty.

**Key Classes**:
- `ChemNetSemanticEncoder`: Wrapper for fcd_torch that extracts 512-dim embeddings
  - `get_embeddings(smiles_list)`: Returns [N, 512] embedding array

**Key Functions**:
- `mol_semantic_vector()`: Converts generated (x, adj) tensors → SMILES → embeddings
- `mol_semantic_generative_uncertainty()`: Computes uncertainty for a batch (Eq. 8)
- `mol_semantic_per_graph_uncertainty()`: Computes per-molecule uncertainty
- `semantic_uncertainty_trace()`: Uncertainty = trace(Cov(embeddings))
- `semantic_uncertainty_entropy()`: Uncertainty = entropy of Gaussian approximation

**Algorithm (Jazbec et al. Algorithm 1)**:
```python
# Generate FIXED noise z (same for all posterior samples)
z_x = sde_x.prior_sampling(shape)
z_adj = sde_adj.prior_sampling_sym(shape)

for m in range(M):  # M posterior samples
    # Sample parameters from posterior
    theta_m = laplace.sample_parameters()

    # Generate molecule with FIXED noise, varying parameters
    mol_m = generate(z_x, z_adj, theta_m)

    # Get semantic embedding
    emb_m = chemnet.embed(mol_m)

# Compute uncertainty as variability in embedding space
uncertainty = trace(Cov(emb_1, ..., emb_M))
```

### 3. `evaluate_uncertainty_filtering.py` - Main Evaluation

**Purpose**: Reproduces Figure 3 from Jazbec et al. - comparing uncertainty-based filtering vs random selection.

**Pipeline**:
1. Load pretrained GDSS model
2. Create Laplace approximation and fit Fisher
3. Generate N molecules with per-molecule uncertainty estimates
4. Evaluate metrics at different filtering levels (keep top-k by low uncertainty)
5. Compare against random baseline
6. Generate plots

**Metrics computed**:
- **Validity**: % of chemically valid molecules
- **Uniqueness**: % of unique molecules among valid
- **Novelty**: % not in training set
- **FCD**: Fréchet ChemNet Distance (lower = better)

### 4. `run_mol_semantic_uncertainty.py` - Simple Pipeline

**Purpose**: Quick testing of uncertainty computation on small samples.

Simpler version of the evaluation script for debugging and verification.

---

## How to Run

### Prerequisites

```bash
# Install dependencies
pip install torch numpy scipy matplotlib tqdm rdkit
pip install laplace-torch  # For Laplace approximation
pip install fcd_torch>=1.0.5  # For ChemNet embeddings

# Preprocess data (if not done)
python data/preprocess.py --dataset ZINC250k
```

### Download Pretrained Models

Place pretrained GDSS checkpoints in:
```
checkpoints/ZINC250k/gdss_ZINC250k.pth
checkpoints/QM9/gdss_QM9.pth
```

### Run Evaluation (GPU Server - Recommended)

```bash
# Full evaluation with 500 molecules
python evaluate_uncertainty_filtering.py \
    --dataset ZINC250k \
    --n_molecules 500 \
    --batch_size 32 \
    --n_steps 1000 \
    --n_posterior 10 \
    --num_fit_batches 20 \
    --seed 42 \
    --output_dir ./results

# Quick test with fewer samples
python evaluate_uncertainty_filtering.py \
    --dataset ZINC250k \
    --n_molecules 100 \
    --batch_size 16 \
    --n_steps 100 \
    --n_posterior 5 \
    --num_fit_batches 5
```

### Run Simple Pipeline

```bash
python run_mol_semantic_uncertainty.py \
    --dataset ZINC250k \
    --n_samples 10 \
    --n_posterior 20 \
    --n_steps 100
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name (ZINC250k, QM9) | ZINC250k |
| `--n_molecules` | Total molecules to generate | 500 |
| `--batch_size` | Batch size for generation | 32 |
| `--n_steps` | Diffusion steps (higher = better quality) | 1000 |
| `--n_posterior` | Posterior samples for uncertainty (M) | 10 |
| `--num_fit_batches` | Batches for Fisher fitting | 20 |
| `--prior_precision` | Laplace prior precision | 1.0 |
| `--seed` | Random seed | 42 |
| `--output_dir` | Directory for results | ./results |
| `--save_laplace` | Save fitted Laplace state | False |
| `--load_laplace` | Load pre-fitted Laplace state | None |

---

## Expected Runtime (GPU vs CPU)

| Setting | GPU (NVIDIA) | CPU |
|---------|--------------|-----|
| Fisher fitting (20 batches) | ~2 min | ~10 min |
| Generate 100 molecules | ~3 min | ~15 min |
| Generate 500 molecules | ~15 min | ~60 min |
| Full evaluation (500 mol) | ~20 min | ~90 min |

---

## Output Files

After running `evaluate_uncertainty_filtering.py`:

```
results/
├── filtering_results_ZINC250k.png    # Figure 3-style comparison plot
├── uncertainty_correlation_ZINC250k.png  # Uncertainty vs quality analysis
```

---

## Key Concepts

### Why Fixed Noise?

The Jazbec et al. method measures **epistemic uncertainty** (model uncertainty) by:
1. Fixing the random noise z that initializes the diffusion process
2. Sampling different parameter sets θ from the posterior
3. Generating multiple outputs from the SAME noise with DIFFERENT parameters
4. Measuring how much the outputs vary → this is the uncertainty

If we used different noise for each sample, we'd be measuring aleatoric uncertainty (noise) instead of epistemic uncertainty (model knowledge).

### Last-Layer Laplace

Instead of approximating the full posterior over all parameters (expensive), we:
1. Only consider the last layer of the neural network
2. Use diagonal Fisher approximation (assumes independence between parameters)
3. This gives us: q(θ) ≈ N(θ_MAP, H^{-1}) where H is the Fisher information

### Semantic Likelihood (ChemNet)

ChemNet is a neural network trained on ~6000 biological assays. Its penultimate layer produces 512-dimensional embeddings that capture both:
- Chemical structure
- Biological activity

By measuring uncertainty in this space (rather than raw molecular space), we get semantically meaningful uncertainty estimates.

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size`
- Reduce `--n_posterior`

### Poor Quality Molecules
- Increase `--n_steps` (1000 is standard for ZINC250k)
- Check that checkpoint is correct

### Slow Fisher Fitting
- Reduce `--num_fit_batches` (5-10 is usually sufficient)
- Use GPU

### FCD Computation Fails
- Ensure at least 10 valid molecules
- Check fcd_torch installation

---

## Citation

If you use this code, please cite:

```bibtex
@article{jazbec2025generative,
  title={Generative Uncertainty in Diffusion Models},
  author={Jazbec, Metod and others},
  year={2025}
}

@article{jo2022gdss,
  title={Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations},
  author={Jo, Jaehyeong and others},
  year={2022}
}
```
