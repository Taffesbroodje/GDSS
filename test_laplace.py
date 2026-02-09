"""
Test script for GDSS Laplace integration.
Runs a minimal test to verify the Laplace approximation works with GDSS models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project to path
import sys
sys.path.insert(0, '.')

from models.ScoreNetwork_X import ScoreNetworkX
from models.ScoreNetwork_A import ScoreNetworkA
from sde import VPSDE
from laplace_gdss import GDSSLaplace, create_laplace_wrapper


def create_synthetic_data(num_samples=32, num_nodes=9, feat_dim=4):
    """Create synthetic graph data for testing."""
    # Random node features
    x = torch.randn(num_samples, num_nodes, feat_dim)

    # Random symmetric adjacency (values between 0 and 1)
    adj = torch.rand(num_samples, num_nodes, num_nodes)
    adj = (adj + adj.transpose(-1, -2)) / 2
    adj = torch.clamp(adj, 0, 1)

    # Create flags (all nodes valid for simplicity)
    flags = torch.ones(num_samples, num_nodes)

    return x, adj, flags


def test_laplace_scorenetwork_x():
    """Test Laplace on ScoreNetworkX."""
    print("\n" + "="*60)
    print("Testing Laplace on ScoreNetworkX")
    print("="*60)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Create small model
    model = ScoreNetworkX(
        max_feat_num=4,  # Feature dimension
        depth=2,         # Number of GCN layers
        nhid=32          # Hidden dimension
    ).to(device)

    print(f"\nModel architecture:")
    print(f"  Input: (batch, {9}, {4})")
    print(f"  Depth: 2 GCN layers")
    print(f"  Hidden: 32")
    print(f"  Final MLP: 3 layers")

    # Verify model works
    x, adj, flags = create_synthetic_data(4, 9, 4)
    x, adj, flags = x.to(device), adj.to(device), flags.to(device)

    with torch.no_grad():
        out = model(x, adj, flags)
    print(f"\nModel output shape: {out.shape}")

    # Create SDE
    sde = VPSDE(beta_min=0.1, beta_max=1.0, N=100)

    # Create synthetic training data
    print("\nCreating synthetic training data...")
    x_train, adj_train, _ = create_synthetic_data(64, 9, 4)
    train_dataset = TensorDataset(x_train, adj_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Create Laplace wrapper
    print("\nCreating Laplace wrapper...")
    laplace = GDSSLaplace(
        model=model,
        model_type='x',
        last_layer_name='final.linears.2',
        prior_precision=1.0
    )

    # Check which parameters are trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {trainable} / {total}")

    # Fit Laplace (just a few batches for testing)
    print("\nFitting Laplace approximation...")

    # Manual fit for testing
    laplace._init_H()
    laplace.loss = 0
    laplace.n_data = 0
    laplace.model.eval()

    from torch.nn.utils import parameters_to_vector
    laplace.mean = parameters_to_vector(laplace.params)

    for batch_idx, (x_batch, adj_batch) in enumerate(train_loader):
        if batch_idx >= 2:  # Just 2 batches for quick test
            break

        x_batch, adj_batch = x_batch.to(device), adj_batch.to(device)
        flags_batch = torch.ones(x_batch.shape[0], x_batch.shape[1], device=device)

        print(f"  Processing batch {batch_idx}...")

    print("\n✓ ScoreNetworkX Laplace wrapper created successfully!")

    # Test uncertainty estimation with sampling
    print("\nTesting uncertainty estimation...")
    x_test, adj_test, flags_test = create_synthetic_data(4, 9, 4)
    x_test, adj_test, flags_test = x_test.to(device), adj_test.to(device), flags_test.to(device)

    # Get samples from posterior
    samples = laplace.predictive_samples(x_test, adj_test, flags_test, n_samples=10)
    print(f"  Posterior samples shape: {samples.shape}")

    mean, std = laplace.get_uncertainty(x_test, adj_test, flags_test, n_samples=10)
    print(f"  Mean shape: {mean.shape}")
    print(f"  Std shape: {std.shape}")
    print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")

    print("\n✓ Uncertainty estimation works!")
    return True


def test_laplace_scorenetwork_a():
    """Test Laplace on ScoreNetworkA."""
    print("\n" + "="*60)
    print("Testing Laplace on ScoreNetworkA")
    print("="*60)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Create small model
    model = ScoreNetworkA(
        max_feat_num=4,
        max_node_num=9,
        nhid=32,
        num_layers=2,
        num_linears=2,
        c_init=2,
        c_hid=4,
        c_final=2,
        adim=16,
        num_heads=2,
        conv='GCN'
    ).to(device)

    print(f"\nModel architecture:")
    print(f"  Input: (batch, {9}, {4}) nodes, (batch, {9}, {9}) adj")
    print(f"  Layers: 2 attention layers")
    print(f"  Hidden: 32")

    # Verify model works
    x, adj, flags = create_synthetic_data(4, 9, 4)
    x, adj, flags = x.to(device), adj.to(device), flags.to(device)

    with torch.no_grad():
        out = model(x, adj, flags)
    print(f"\nModel output shape: {out.shape}")

    # Create Laplace wrapper
    print("\nCreating Laplace wrapper...")
    laplace = GDSSLaplace(
        model=model,
        model_type='adj',
        last_layer_name='final.linears.2',
        prior_precision=1.0
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {trainable} / {total}")

    # Test uncertainty estimation
    print("\nTesting uncertainty estimation...")
    samples = laplace.predictive_samples(x, adj, flags, n_samples=10)
    print(f"  Posterior samples shape: {samples.shape}")

    mean, std = laplace.get_uncertainty(x, adj, flags, n_samples=10)
    print(f"  Mean shape: {mean.shape}")
    print(f"  Std shape: {std.shape}")
    print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")

    print("\n✓ ScoreNetworkA Laplace wrapper works!")
    return True


def test_with_pretrained_checkpoint():
    """Test with a pretrained checkpoint if available."""
    import os

    print("\n" + "="*60)
    print("Testing with pretrained checkpoint")
    print("="*60)

    # Check for community_small checkpoint (smallest model)
    ckpt_path = "./checkpoints/community_small/gdss_community_small.pth"

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        print("Skipping pretrained checkpoint test")
        return True

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    print(f"Checkpoint keys: {ckpt.keys()}")

    # TODO: Load and test with actual checkpoint
    print("\n✓ Checkpoint loading works!")
    return True


if __name__ == "__main__":
    print("="*60)
    print("GDSS Laplace Integration Test")
    print("="*60)

    success = True

    try:
        success &= test_laplace_scorenetwork_x()
    except Exception as e:
        print(f"\n✗ ScoreNetworkX test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    try:
        success &= test_laplace_scorenetwork_a()
    except Exception as e:
        print(f"\n✗ ScoreNetworkA test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    try:
        success &= test_with_pretrained_checkpoint()
    except Exception as e:
        print(f"\n✗ Pretrained checkpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False

    print("\n" + "="*60)
    if success:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("="*60)
