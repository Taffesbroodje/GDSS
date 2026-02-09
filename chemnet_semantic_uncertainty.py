"""
ChemNet-based Semantic Uncertainty for Molecular Generation in GDSS.

This module implements the Jazbec et al. (2025) "Generative Uncertainty in
Diffusion Models" framework for molecular generation, using ChemNet embeddings
as the semantic likelihood (analogous to CLIP embeddings for images).

Key components:
- ChemNetSemanticEncoder: Extracts 512-dim embeddings from SMILES
- mol_semantic_generative_uncertainty: Computes uncertainty via posterior sampling

Reference:
    Jazbec et al. "Generative Uncertainty in Diffusion Models" (2025)
    Preuer et al. "Fréchet ChemNet Distance" (2018) - ChemNet architecture
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from rdkit import Chem

from utils.mol_utils import gen_mol, mols_to_smiles


class ChemNetSemanticEncoder:
    """
    Semantic encoder using ChemNet embeddings for molecules.

    ChemNet is trained on ~6,000 biological assays and captures both
    chemical structure and biological activity properties.

    Output dimensionality: 512
    """

    def __init__(self, device: str = "cpu", n_jobs: int = 1, batch_size: int = 512):
        """
        Initialize the ChemNet encoder.

        Parameters
        ----------
        device : str
            Device for ChemNet inference ('cpu', 'cuda', 'mps')
        n_jobs : int
            Number of parallel jobs for SMILES parsing
        batch_size : int
            Batch size for ChemNet inference
        """
        self.device = device
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self._fcd = None

    @property
    def fcd(self):
        """Lazy initialization of FCD model."""
        if self._fcd is None:
            try:
                from fcd_torch import FCD
                # Note: fcd_torch may not support MPS, fallback to CPU
                device = self.device
                if device == "mps":
                    device = "cpu"
                self._fcd = FCD(device=device, n_jobs=self.n_jobs, batch_size=self.batch_size)
            except ImportError:
                raise ImportError(
                    "fcd_torch is required for ChemNet embeddings. "
                    "Install with: pip install fcd_torch>=1.0.5"
                )
        return self._fcd

    def get_embeddings(self, smiles_list: List[str]) -> np.ndarray:
        """
        Get ChemNet embeddings for a list of SMILES strings.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings

        Returns
        -------
        embeddings : np.ndarray, shape [N, 512]
            ChemNet penultimate layer activations
        """
        if len(smiles_list) == 0:
            return np.zeros((0, 512), dtype=np.float32)

        # Filter out invalid SMILES
        valid_smiles = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            if smi is not None and len(smi) > 0:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    valid_smiles.append(smi)
                    valid_indices.append(i)

        if len(valid_smiles) == 0:
            return np.zeros((len(smiles_list), 512), dtype=np.float32)

        # Get embeddings using fcd_torch
        embeddings = self.fcd.get_predictions(valid_smiles)

        # Create output array with zeros for invalid molecules
        output = np.zeros((len(smiles_list), 512), dtype=np.float32)
        for i, idx in enumerate(valid_indices):
            output[idx] = embeddings[i]

        return output

    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        return 512


def mol_semantic_vector(
    x: torch.Tensor,
    adj: torch.Tensor,
    dataset: str,
    encoder: ChemNetSemanticEncoder,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute ChemNet semantic embeddings for generated molecules.

    Parameters
    ----------
    x : torch.Tensor, shape [B, N, F]
        Node features (atom types), continuous from diffusion
    adj : torch.Tensor, shape [B, N, N]
        Adjacency matrices (bond types), continuous from diffusion
    dataset : str
        Dataset name ('QM9' or 'ZINC250k')
    encoder : ChemNetSemanticEncoder
        ChemNet encoder instance

    Returns
    -------
    embeddings : np.ndarray, shape [B, 512]
        ChemNet embeddings for each molecule
    smiles : List[str]
        SMILES strings for each molecule (None for invalid)
    """
    from utils.graph_utils import quantize_mol

    B, N, F = x.shape

    # Quantize adjacency matrix for molecules
    # quantize_mol returns {0,1,2,3} where: 0=no bond, 1=single, 2=double, 3=triple
    samples_int = quantize_mol(adj.detach().cpu())

    # Convert bond types: 0,1,2,3 -> 3,0,1,2 (no bond becomes channel 3)
    samples_int = samples_int - 1
    samples_int[samples_int == -1] = 3

    # One-hot encode adjacency: [B,N,N] -> [B,4,N,N]
    adj_4ch = torch.nn.functional.one_hot(
        torch.tensor(samples_int, dtype=torch.long), num_classes=4
    ).permute(0, 3, 1, 2).float()

    # Binarize node features and add virtual node feature
    x_binary = torch.where(x.detach().cpu() > 0.5, 1.0, 0.0)
    # Add virtual node feature (1 - sum of other features)
    virtual_feat = 1.0 - x_binary.sum(dim=-1, keepdim=True)
    x_with_virtual = torch.cat([x_binary, virtual_feat], dim=-1)

    # Generate molecules
    mols, _ = gen_mol(x_with_virtual, adj_4ch, dataset, largest_connected_comp=True)

    # Convert to SMILES
    smiles_list = []
    for i in range(B):
        if i < len(mols) and mols[i] is not None:
            try:
                smi = Chem.MolToSmiles(mols[i])
                smiles_list.append(smi)
            except:
                smiles_list.append(None)
        else:
            smiles_list.append(None)

    # Pad smiles_list if some molecules were filtered
    while len(smiles_list) < B:
        smiles_list.append(None)

    # Get ChemNet embeddings
    embeddings = encoder.get_embeddings(smiles_list)

    return embeddings, smiles_list


def semantic_uncertainty_trace(embeddings: np.ndarray) -> float:
    """
    Compute semantic uncertainty as trace of covariance matrix.

    This is the total variance across all embedding dimensions,
    following Jazbec et al. (2025).

    Parameters
    ----------
    embeddings : np.ndarray, shape [M, D]
        M samples of D-dimensional embeddings

    Returns
    -------
    uncertainty : float
        Trace of covariance matrix (sum of variances)
    """
    if embeddings.shape[0] < 2:
        return 0.0

    # Filter out zero embeddings (invalid molecules)
    valid_mask = np.any(embeddings != 0, axis=1)
    valid_embeddings = embeddings[valid_mask]

    if valid_embeddings.shape[0] < 2:
        return float('inf')  # High uncertainty if most samples invalid

    centered = valid_embeddings - valid_embeddings.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    return float(np.trace(cov))


def semantic_uncertainty_entropy(
    embeddings: np.ndarray,
    sigma_squared: float = 1e-3,
) -> float:
    """
    Compute semantic uncertainty as entropy of Gaussian mixture.

    Following Jazbec et al. (2025), this computes the entropy of a
    Gaussian with diagonal covariance estimated from the samples,
    plus observation noise.

    H(p) = 0.5 * D * (1 + log(2π)) + 0.5 * sum(log(σ²_i + σ²_obs))

    Parameters
    ----------
    embeddings : np.ndarray, shape [M, D]
        M samples of D-dimensional embeddings
    sigma_squared : float
        Observation noise variance

    Returns
    -------
    entropy : float
        Entropy of the Gaussian approximation
    """
    if embeddings.shape[0] < 2:
        return 0.0

    # Filter out zero embeddings (invalid molecules)
    valid_mask = np.any(embeddings != 0, axis=1)
    valid_embeddings = embeddings[valid_mask]

    if valid_embeddings.shape[0] < 2:
        return float('inf')

    D = valid_embeddings.shape[1]

    # Compute variance along each dimension
    var = np.var(valid_embeddings, axis=0)

    # Add observation noise
    var_with_noise = var + sigma_squared

    # Compute entropy of diagonal Gaussian
    entropy = 0.5 * D * (1 + np.log(2 * np.pi)) + 0.5 * np.sum(np.log(var_with_noise))

    return float(entropy)


@torch.no_grad()
def mol_semantic_generative_uncertainty(
    model_x,
    model_adj,
    laplace_x,
    laplace_adj,
    sde_x,
    sde_adj,
    init_flags: torch.Tensor,
    shape_x: Tuple[int, int, int],
    shape_adj: Tuple[int, int, int],
    dataset: str = "ZINC250k",
    n_steps: int = 100,
    n_posterior: int = 20,
    seed: int = 0,
    device: str = "cpu",
    pick_index: Optional[int] = None,
    uncertainty_type: str = "trace",
    sigma_squared: float = 1e-3,
) -> Tuple[float, np.ndarray, List[List[str]]]:
    """
    Compute molecular semantic generative uncertainty following Jazbec et al.

    CRITICAL: For a FIXED noise z, we:
      1. Sample M parameter sets {θ1, ..., θM} from the Laplace posterior
      2. Generate M molecules using the SAME fixed noise z
      3. Extract ChemNet embeddings for each
      4. Compute uncertainty as entropy/trace of embedding distribution

    This is the correct implementation of Eq. 8 from Jazbec et al. (2025).

    Parameters
    ----------
    model_x, model_adj : nn.Module
        Score networks for node features and adjacency
    laplace_x, laplace_adj : GDSSLaplaceFull
        Laplace approximation wrappers
    sde_x, sde_adj : SDE
        SDEs for diffusion process
    init_flags : torch.Tensor
        Initialization flags for graph sizes
    shape_x, shape_adj : Tuple
        Shapes for generated tensors
    dataset : str
        Dataset name ('QM9' or 'ZINC250k')
    n_steps : int
        Number of diffusion steps
    n_posterior : int
        Number of posterior parameter samples (M in the paper)
    seed : int
        Random seed for the FIXED noise z
    device : str
        Device for computation
    pick_index : Optional[int]
        If specified, only compute uncertainty for this molecule index.
        If None, compute for all molecules.
    uncertainty_type : str
        'trace' for covariance trace, 'entropy' for Gaussian entropy
    sigma_squared : float
        Observation noise for entropy computation (σ² in Eq. 7)

    Returns
    -------
    uncertainty : float
        Semantic generative uncertainty u(z)
    embeddings : np.ndarray, shape [M, B, 512]
        ChemNet embeddings from each posterior sample
    smiles_list : List[List[str]]
        SMILES strings from each posterior sample
    """
    from laplace_gdss_full import uncertainty_aware_sampling

    # Set seed for FIXED noise z (this is crucial for Jazbec-style uncertainty)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate FIXED initial noise z - this is the key difference!
    # All posterior samples will use this same noise
    fixed_noise_x = sde_x.prior_sampling(shape_x)
    fixed_noise_adj = sde_adj.prior_sampling_sym(shape_adj)

    # Initialize ChemNet encoder
    encoder = ChemNetSemanticEncoder(device=device)

    # Save MAP parameters
    theta_x_map = laplace_x._get_param_vector().clone()
    theta_a_map = laplace_adj._get_param_vector().clone()

    all_embeddings = []
    all_smiles = []

    for i in range(n_posterior):
        # Sample posterior parameters θm ~ q(θ|D)
        theta_x = laplace_x.sample_parameters(1)[0]
        theta_a = laplace_adj.sample_parameters(1)[0]

        laplace_x._set_param_vector(theta_x)
        laplace_adj._set_param_vector(theta_a)

        # Disable uncertainty logic during generation
        laplace_x.fitted = False
        laplace_adj.fitted = False

        # Generate with FIXED noise z (same for all posterior samples!)
        x_gen, adj_gen, _ = uncertainty_aware_sampling(
            model_x, model_adj,
            laplace_x, laplace_adj,
            sde_x, sde_adj,
            init_flags,
            shape_x, shape_adj,
            n_steps=n_steps,
            n_uncertainty_samples=0,
            device=device,
            fixed_noise_x=fixed_noise_x,  # FIXED noise!
            fixed_noise_adj=fixed_noise_adj,  # FIXED noise!
        )

        # Restore fitted flags
        laplace_x.fitted = True
        laplace_adj.fitted = True

        # Get ChemNet embeddings em = c_φ(g_θm(z))
        embeddings, smiles = mol_semantic_vector(x_gen, adj_gen, dataset, encoder)

        all_embeddings.append(embeddings)
        all_smiles.append(smiles)

    # Restore MAP parameters
    laplace_x._set_param_vector(theta_x_map)
    laplace_adj._set_param_vector(theta_a_map)
    laplace_x.fitted = True
    laplace_adj.fitted = True

    if len(all_embeddings) < 2:
        return 0.0, np.zeros((0, 512)), []

    all_embeddings = np.stack(all_embeddings, axis=0)  # [M, B, 512]

    # Compute uncertainty u(z) = H(p(x|z, D)) following Eq. 8
    if pick_index is not None:
        # Uncertainty for a single molecule
        idx = min(pick_index, all_embeddings.shape[1] - 1)
        emb = all_embeddings[:, idx, :]  # [M, 512]

        if uncertainty_type == "trace":
            u = semantic_uncertainty_trace(emb)
        else:
            u = semantic_uncertainty_entropy(emb, sigma_squared)
    else:
        # Average uncertainty across all molecules in batch
        B = all_embeddings.shape[1]
        uncertainties = []
        for b in range(B):
            emb = all_embeddings[:, b, :]  # [M, 512]
            if uncertainty_type == "trace":
                u_b = semantic_uncertainty_trace(emb)
            else:
                u_b = semantic_uncertainty_entropy(emb, sigma_squared)
            uncertainties.append(u_b)
        u = np.mean([u for u in uncertainties if np.isfinite(u)])

    return u, all_embeddings, all_smiles


@torch.no_grad()
def mol_semantic_per_graph_uncertainty(
    model_x,
    model_adj,
    laplace_x,
    laplace_adj,
    sde_x,
    sde_adj,
    init_flags: torch.Tensor,
    shape_x: Tuple[int, int, int],
    shape_adj: Tuple[int, int, int],
    dataset: str = "ZINC250k",
    n_steps: int = 100,
    n_posterior: int = 20,
    seed: int = 0,
    device: str = "cpu",
    uncertainty_type: str = "trace",
    sigma_squared: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
    """
    Compute per-molecule semantic uncertainty following Jazbec et al. (2025).

    For each molecule position b in the batch, computes u(z_b) using FIXED
    noise z_b and M posterior parameter samples.

    Parameters
    ----------
    (Same as mol_semantic_generative_uncertainty)

    Returns
    -------
    uncertainties : np.ndarray, shape [B]
        Semantic uncertainty u(z_b) for each molecule
    embeddings : np.ndarray, shape [M, B, 512]
        ChemNet embeddings from each posterior sample
    smiles_list : List[List[str]]
        SMILES from each posterior sample
    """
    from laplace_gdss_full import uncertainty_aware_sampling

    # Set seed for FIXED noise z
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate FIXED initial noise - same for all posterior samples!
    fixed_noise_x = sde_x.prior_sampling(shape_x)
    fixed_noise_adj = sde_adj.prior_sampling_sym(shape_adj)

    encoder = ChemNetSemanticEncoder(device=device)

    theta_x_map = laplace_x._get_param_vector().clone()
    theta_a_map = laplace_adj._get_param_vector().clone()

    all_embeddings = []
    all_smiles = []

    for i in range(n_posterior):
        # Sample posterior parameters θm ~ q(θ|D)
        theta_x = laplace_x.sample_parameters(1)[0]
        theta_a = laplace_adj.sample_parameters(1)[0]

        laplace_x._set_param_vector(theta_x)
        laplace_adj._set_param_vector(theta_a)

        laplace_x.fitted = False
        laplace_adj.fitted = False

        # Generate with FIXED noise (crucial for correct uncertainty!)
        x_gen, adj_gen, _ = uncertainty_aware_sampling(
            model_x, model_adj,
            laplace_x, laplace_adj,
            sde_x, sde_adj,
            init_flags,
            shape_x, shape_adj,
            n_steps=n_steps,
            n_uncertainty_samples=0,
            device=device,
            fixed_noise_x=fixed_noise_x,  # FIXED noise!
            fixed_noise_adj=fixed_noise_adj,  # FIXED noise!
        )

        laplace_x.fitted = True
        laplace_adj.fitted = True

        embeddings, smiles = mol_semantic_vector(x_gen, adj_gen, dataset, encoder)
        all_embeddings.append(embeddings)
        all_smiles.append(smiles)

    laplace_x._set_param_vector(theta_x_map)
    laplace_adj._set_param_vector(theta_a_map)
    laplace_x.fitted = True
    laplace_adj.fitted = True

    if len(all_embeddings) < 2:
        B = shape_x[0]
        return np.full(B, float('inf')), np.zeros((0, B, 512)), []

    all_embeddings = np.stack(all_embeddings, axis=0)  # [M, B, 512]
    B = all_embeddings.shape[1]

    # Compute per-molecule uncertainty u(z_b) for each molecule b
    uncertainties = np.zeros(B)
    for b in range(B):
        emb = all_embeddings[:, b, :]  # [M, 512]
        if uncertainty_type == "trace":
            uncertainties[b] = semantic_uncertainty_trace(emb)
        else:
            uncertainties[b] = semantic_uncertainty_entropy(emb, sigma_squared)

    return uncertainties, all_embeddings, all_smiles


def analyze_uncertainty_quality(
    uncertainties: np.ndarray,
    smiles_list: List[List[str]],
    reference_smiles: Optional[List[str]] = None,
) -> dict:
    """
    Analyze the quality of uncertainty estimates.

    Parameters
    ----------
    uncertainties : np.ndarray, shape [B]
        Per-molecule uncertainties
    smiles_list : List[List[str]]
        SMILES from each posterior sample [M][B]
    reference_smiles : Optional[List[str]]
        Reference SMILES for novelty/validity analysis

    Returns
    -------
    analysis : dict
        Statistics about uncertainty and validity
    """
    B = len(uncertainties)
    M = len(smiles_list)

    # Count valid molecules per position
    valid_counts = np.zeros(B)
    for m in range(M):
        for b in range(B):
            if smiles_list[m][b] is not None:
                valid_counts[b] += 1

    validity_rate = valid_counts / M

    # Correlation between uncertainty and validity
    finite_mask = np.isfinite(uncertainties)
    if finite_mask.sum() > 1:
        corr = np.corrcoef(uncertainties[finite_mask], validity_rate[finite_mask])[0, 1]
    else:
        corr = 0.0

    return {
        "mean_uncertainty": float(np.nanmean(uncertainties)),
        "std_uncertainty": float(np.nanstd(uncertainties)),
        "min_uncertainty": float(np.nanmin(uncertainties)),
        "max_uncertainty": float(np.nanmax(uncertainties)),
        "mean_validity_rate": float(np.mean(validity_rate)),
        "uncertainty_validity_correlation": float(corr),
        "n_infinite_uncertainty": int((~finite_mask).sum()),
    }
