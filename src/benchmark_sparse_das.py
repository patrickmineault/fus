"""
Verification and benchmarking script for sparse matrix DAS implementation.

This script:
1. Verifies that the sparse matrix version produces equivalent results to the reference
2. Benchmarks the sparse matrix version against naive and optimized implementations
   for 1, 2, 4, and 8 images in parallel/sequence
"""

import time

import numpy as np

from ultrasound_processing import (
    SimParams,
    beamform_das,
    beamform_das_sparse_with_shape,
    beamform_das_vectorized,
    build_das_sparse_matrix,
    make_image_grid,
    simulate_forward_channels,
)


def verify_equivalence():
    """Verify that sparse matrix version produces equivalent results."""
    print("=" * 70)
    print("VERIFICATION: Comparing sparse matrix vs reference implementation")
    print("=" * 70)

    # Create a simple test case
    P = SimParams()
    P.angles_deg = (-4, 0, 4)  # Use fewer angles for faster testing

    # Create a simple phantom with a few point scatterers
    np.random.seed(42)
    n_scatterers = 50
    x_span = P.x_span
    z_span = P.z_span
    scat_pos = np.random.rand(n_scatterers, 2)
    scat_pos[:, 0] = scat_pos[:, 0] * (x_span[1] - x_span[0]) + x_span[0]
    scat_pos[:, 1] = scat_pos[:, 1] * (z_span[1] - z_span[0]) + z_span[0]
    scat_amp = np.random.randn(n_scatterers)

    # Simulate forward channels
    print("Simulating forward channels...")
    Y, times, elem_pos, betas, _, _ = simulate_forward_channels(P, scat_pos, scat_amp)
    print(f"  Y shape: {Y.shape}")

    # Create image grid
    dx = 200e-6  # Use coarser grid for faster testing
    dz = 200e-6
    x, z, X_grid, Z_grid = make_image_grid(P.x_span, P.z_span, dx, dz)
    print(f"  Image grid: {len(z)} x {len(x)}")

    # Compute omega
    omega = 2 * np.pi * P.f_carrier

    # 1. Reference implementation (alpha=0)
    print("\nRunning reference implementation...")
    t0 = time.time()
    img_ref = beamform_das(
        Y, times, elem_pos, x, z, P.c, betas, omega, alpha_np_per_m=0.0
    )
    t_ref = time.time() - t0
    print(f"  Time: {t_ref:.3f}s")
    print(f"  Image shape: {img_ref.shape}")

    # 2. Build sparse matrix
    print("\nBuilding sparse projection matrix...")
    t0 = time.time()
    X_sparse = build_das_sparse_matrix(times, elem_pos, x, z, P.c, betas, omega)
    t_build = time.time() - t0
    print(f"  Time to build: {t_build:.3f}s")
    print(f"  Matrix shape: {X_sparse.shape}")
    print(f"  Non-zero elements: {X_sparse.nnz:,}")
    print(
        f"  Sparsity: {X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]) * 100:.2f}%"
    )
    print(f"  Memory (approx): {X_sparse.data.nbytes / 1e6:.1f} MB")

    # 3. Sparse matrix implementation
    print("\nRunning sparse matrix implementation...")
    t0 = time.time()
    img_sparse = beamform_das_sparse_with_shape(Y, X_sparse, len(z), len(x))
    t_sparse = time.time() - t0
    print(f"  Time: {t_sparse:.3f}s")
    print(f"  Image shape: {img_sparse.shape}")

    # 4. Compare results
    print("\nComparing results...")
    abs_diff = np.abs(img_sparse - img_ref)
    rel_diff = abs_diff / (np.abs(img_ref) + 1e-10)
    print(f"  Max absolute difference: {abs_diff.max():.2e}")
    print(f"  Mean absolute difference: {abs_diff.mean():.2e}")
    print(f"  Max relative difference: {rel_diff.max():.2e}")
    print(f"  Mean relative difference: {rel_diff.mean():.2e}")

    # Check if they're close
    if np.allclose(img_sparse, img_ref, rtol=1e-5, atol=1e-10):
        print("  ✓ PASSED: Results are equivalent!")
    else:
        print("  ✗ FAILED: Results differ significantly!")
        return False

    return True


def benchmark_implementations(n_images_list=[1, 2, 4, 8]):
    """Benchmark sparse vs naive vs optimized for multiple images."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Comparing performance for multiple images")
    print("=" * 70)

    # Create test case
    P = SimParams()
    P.angles_deg = (-8, -6, -4, -2, 0, 2, 4, 6, 8)  # Full set of angles

    # Create phantom
    np.random.seed(42)
    n_scatterers = 100
    x_span = P.x_span
    z_span = P.z_span
    scat_pos = np.random.rand(n_scatterers, 2)
    scat_pos[:, 0] = scat_pos[:, 0] * (x_span[1] - x_span[0]) + x_span[0]
    scat_pos[:, 1] = scat_pos[:, 1] * (z_span[1] - z_span[0]) + z_span[0]
    scat_amp = np.random.randn(n_scatterers)

    # Simulate forward channels
    print("\nSimulating forward channels...")
    Y, times, elem_pos, betas, _, _ = simulate_forward_channels(P, scat_pos, scat_amp)
    print(f"  Y shape: {Y.shape}")

    # Create image grid
    dx = 150e-6
    dz = 150e-6
    x, z, X_grid, Z_grid = make_image_grid(P.x_span, P.z_span, dx, dz)
    print(f"  Image grid: {len(z)} x {len(x)}")

    # Compute omega
    omega = 2 * np.pi * P.f_carrier

    # Build sparse matrix once
    print("\nBuilding sparse projection matrix...")
    t0 = time.time()
    X_sparse = build_das_sparse_matrix(times, elem_pos, x, z, P.c, betas, omega)
    t_build = time.time() - t0
    print(f"  Time to build: {t_build:.3f}s")
    print(f"  Matrix shape: {X_sparse.shape}")
    print(f"  Non-zero elements: {X_sparse.nnz:,}")

    # Benchmark for different numbers of images
    results = {
        "n_images": [],
        "naive_time": [],
        "optimized_time": [],
        "sparse_time": [],
        "sparse_with_build_time": [],
    }

    for n_images in n_images_list:
        print(f"\n--- Benchmarking with {n_images} image(s) ---")

        # Create n_images copies of Y (simulating multiple datasets)
        Y_batch = np.array([Y for _ in range(n_images)])

        # 1. Naive implementation (reference)
        print(f"  Running naive (reference) implementation...")
        t0 = time.time()
        for i in range(n_images):
            img_naive = beamform_das(
                Y_batch[i], times, elem_pos, x, z, P.c, betas, omega, alpha_np_per_m=0.0
            )
        t_naive = time.time() - t0
        print(f"    Time: {t_naive:.3f}s ({t_naive/n_images:.3f}s per image)")

        # 2. Optimized implementation (Numba)
        print(f"  Running optimized (Numba) implementation...")
        # Warm up
        _ = beamform_das_vectorized(
            Y, times, elem_pos, x, z, P.c, betas, omega, alpha_np_per_m=0.0
        )
        t0 = time.time()
        for i in range(n_images):
            img_opt = beamform_das_vectorized(
                Y_batch[i], times, elem_pos, x, z, P.c, betas, omega, alpha_np_per_m=0.0
            )
        t_opt = time.time() - t0
        print(f"    Time: {t_opt:.3f}s ({t_opt/n_images:.3f}s per image)")

        # 3. Sparse matrix implementation (without build time)
        print(f"  Running sparse matrix implementation...")
        t0 = time.time()
        for i in range(n_images):
            img_sparse = beamform_das_sparse_with_shape(
                Y_batch[i], X_sparse, len(z), len(x)
            )
        t_sparse = time.time() - t0
        print(f"    Time: {t_sparse:.3f}s ({t_sparse/n_images:.3f}s per image)")

        # 4. Sparse matrix with build time amortized
        t_sparse_with_build = t_build + t_sparse
        print(
            f"    Time (with build): {t_sparse_with_build:.3f}s ({t_sparse_with_build/n_images:.3f}s per image)"
        )

        # Store results
        results["n_images"].append(n_images)
        results["naive_time"].append(t_naive)
        results["optimized_time"].append(t_opt)
        results["sparse_time"].append(t_sparse)
        results["sparse_with_build_time"].append(t_sparse_with_build)

        # Speedup analysis
        print(f"    Speedup vs naive: {t_naive/t_sparse:.2f}x")
        print(f"    Speedup vs optimized: {t_opt/t_sparse:.2f}x")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(
        f"{'N Images':<12} {'Naive (s)':<12} {'Optimized (s)':<15} {'Sparse (s)':<12} {'Sparse+Build (s)':<18}"
    )
    print("-" * 70)
    for i in range(len(n_images_list)):
        print(
            f"{results['n_images'][i]:<12} "
            f"{results['naive_time'][i]:<12.3f} "
            f"{results['optimized_time'][i]:<15.3f} "
            f"{results['sparse_time'][i]:<12.3f} "
            f"{results['sparse_with_build_time'][i]:<18.3f}"
        )

    # Speedup table
    print("\n" + "=" * 70)
    print("SPEEDUP TABLE (relative to naive)")
    print("=" * 70)
    print(f"{'N Images':<12} {'Optimized':<15} {'Sparse':<12} {'Sparse+Build':<18}")
    print("-" * 70)
    for i in range(len(n_images_list)):
        print(
            f"{results['n_images'][i]:<12} "
            f"{results['naive_time'][i]/results['optimized_time'][i]:<15.2f}x "
            f"{results['naive_time'][i]/results['sparse_time'][i]:<12.2f}x "
            f"{results['naive_time'][i]/results['sparse_with_build_time'][i]:<18.2f}x"
        )

    return results


if __name__ == "__main__":
    # Step 1: Verify equivalence
    success = verify_equivalence()

    if success:
        # Step 2: Benchmark
        results = benchmark_implementations(n_images_list=[1, 2, 4, 8, 50])

        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("VERIFICATION FAILED - Skipping benchmark")
        print("=" * 70)
