import numpy as np
import torch
import networkx as nx

from utils.graph_utils import quantize, adjs_to_graphs


# ============================================================
# Graph semantic encoder (Option A)
# ============================================================

def graph_semantic_vector(
    G: nx.Graph,
    k_eigs: int = 5,
) -> np.ndarray:
    """
    Compute a deterministic semantic embedding for a graph.

    Parameters
    ----------
    G : networkx.Graph
    k_eigs : int
        Number of Laplacian eigenvalues to include

    Returns
    -------
    v : np.ndarray, shape [D]
        Semantic embedding vector
    """
    if G.number_of_nodes() == 0:
        return np.zeros(8 + k_eigs, dtype=np.float32)

    n = G.number_of_nodes()
    m = G.number_of_edges()

    degrees = np.array([d for _, d in G.degree()], dtype=np.float32)

    mean_deg = degrees.mean() if len(degrees) > 0 else 0.0
    std_deg = degrees.std() if len(degrees) > 0 else 0.0
    max_deg = degrees.max() if len(degrees) > 0 else 0.0

    density = nx.density(G)
    clustering = nx.average_clustering(G) if n > 1 else 0.0

    # Laplacian spectrum (sorted ascending)
    try:
        L = nx.laplacian_matrix(G).toarray()
        eigvals = np.linalg.eigvalsh(L)
        eigvals = np.sort(eigvals)
        eigvals = eigvals[:k_eigs]
        if len(eigvals) < k_eigs:
            eigvals = np.pad(eigvals, (0, k_eigs - len(eigvals)))
        spectral_gap = eigvals[1] if len(eigvals) > 1 else 0.0
    except Exception:
        eigvals = np.zeros(k_eigs, dtype=np.float32)
        spectral_gap = 0.0

    features = np.array([
        n,
        m,
        density,
        mean_deg,
        std_deg,
        max_deg,
        clustering,
        spectral_gap,
    ], dtype=np.float32)

    return np.concatenate([features, eigvals], axis=0)


# ============================================================
# Jazbec-style semantic generative uncertainty
# ============================================================

def semantic_uncertainty_trace(embeddings: np.ndarray) -> float:
    """
    Trace of covariance matrix (total variance).

    embeddings : [M, D]
    """
    if embeddings.shape[0] < 2:
        return 0.0

    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    return float(np.trace(cov))


@torch.no_grad()
def graph_semantic_generative_uncertainty(
    model_x, model_adj,
    laplace_x, laplace_adj,
    sde_x, sde_adj,
    init_flags,
    shape_x, shape_adj,
    n_steps: int = 100,
    n_posterior: int = 20,
    seed: int = 0,
    device: str = "cpu",
    pick_index: int = 0,
):
    """
    Jazbec-style generative uncertainty using graph semantics.

    For a fixed diffusion noise seed:
      - sample last-layer parameters from Laplace posterior
      - generate final graphs
      - embed graphs using graph_semantic_vector
      - compute trace covariance as uncertainty

    Returns
    -------
    u : float
        Semantic generative uncertainty
    vectors : np.ndarray [M, D]
        Semantic embeddings used
    """
    from laplace_gdss_full import uncertainty_aware_sampling

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Save MAP parameters
    theta_x_map = laplace_x._get_param_vector().clone()
    theta_a_map = laplace_adj._get_param_vector().clone()

    vectors = []

    for i in range(n_posterior):
        # Sample posterior parameters
        theta_x = laplace_x.sample_parameters(1)[0]
        theta_a = laplace_adj.sample_parameters(1)[0]

        laplace_x._set_param_vector(theta_x)
        laplace_adj._set_param_vector(theta_a)

        # Disable uncertainty logic during generation
        laplace_x.fitted = False
        laplace_adj.fitted = False

        x_gen, adj_gen, _ = uncertainty_aware_sampling(
            model_x, model_adj,
            laplace_x, laplace_adj,
            sde_x, sde_adj,
            init_flags,
            shape_x, shape_adj,
            n_steps=n_steps,
            n_uncertainty_samples=0,
            device=device,
        )

        # Restore fitted flags
        laplace_x.fitted = True
        laplace_adj.fitted = True

        adj_int = quantize(adj_gen.cpu())
        graphs = adjs_to_graphs(adj_int, True)

        if len(graphs) == 0:
            continue

        idx = min(pick_index, len(graphs) - 1)
        G = graphs[idx]

        vec = graph_semantic_vector(G)
        vectors.append(vec)

    # Restore MAP parameters
    laplace_x._set_param_vector(theta_x_map)
    laplace_adj._set_param_vector(theta_a_map)
    laplace_x.fitted = True
    laplace_adj.fitted = True

    if len(vectors) < 2:
        return 0.0, np.zeros((0, 1))

    vectors = np.stack(vectors, axis=0)
    u = semantic_uncertainty_trace(vectors)

    return u, vectors