"""
Performance benchmarks comparing reference and vectorized implementations.
"""

import time

import numpy as np

from ultrasound_processing import (
    SimParams,
    beamform_das,
    beamform_das_vectorized,
    simulate_forward_channels,
    simulate_forward_channels_vectorized,
)


def sample_scatterers(P: SimParams, n_scatterers=100, seed=42):
    """Helper function to sample random scatterers."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(P.x_span[0], P.x_span[1], n_scatterers)
    zs = rng.uniform(P.z_span[0], P.z_span[1], n_scatterers)
    scat_pos = np.column_stack([xs, zs]).astype(np.float64)
    scat_amp = np.ones(n_scatterers, dtype=np.float64)
    return scat_pos, scat_amp


def make_image_grid(x_span, z_span, dx, dz):
    """Helper to create image grid."""
    x = np.arange(x_span[0], x_span[1] + 1e-12, dx)
    z = np.arange(z_span[0], z_span[1] + 1e-12, dz)
    return x, z


def benchmark_beamforming(n_runs=5):
    """Benchmark beamforming implementations."""
    print("=" * 70)
    print("BEAMFORMING BENCHMARK")
    print("=" * 70)

    # Create test data with realistic size
    P = SimParams()  # Use default parameters from notebook
    scat_pos, scat_amp = sample_scatterers(P, n_scatterers=1000)

    # Simulate signals (using vectorized version for faster setup)
    print("\nGenerating test signals...")
    signals, times, elem_pos, betas, _, _ = simulate_forward_channels_vectorized(
        P, scat_pos, scat_amp
    )

    # Create image grid
    x, z = make_image_grid(P.x_span, P.z_span, 0.1e-3, 0.1e-3)

    # Compute attenuation
    if P.use_attenuation:
        alpha = (P.alpha_db_cm_mhz * (P.f_carrier / 1e6) * 100.0) / 8.686
    else:
        alpha = None

    print(f"Signal shape: {signals.shape}")
    print(f"Image grid: {len(z)} x {len(x)} = {len(z) * len(x)} pixels")
    print(f"Number of angles: {len(betas)}")
    print(f"Number of elements: {len(elem_pos)}")

    # Benchmark reference implementation
    print("\nBenchmarking reference implementation...")
    times_ref = []
    for i in range(n_runs):
        start = time.perf_counter()
        img_ref = beamform_das(signals, times, elem_pos, x, z, P.c, betas, alpha)
        end = time.perf_counter()
        times_ref.append(end - start)
        print(f"  Run {i+1}/{n_runs}: {times_ref[-1]:.4f} s")

    # Benchmark vectorized implementation
    print("\nBenchmarking vectorized implementation...")
    times_vec = []
    for i in range(n_runs):
        start = time.perf_counter()
        img_vec = beamform_das_vectorized(
            signals, times, elem_pos, x, z, P.c, betas, alpha
        )
        end = time.perf_counter()
        times_vec.append(end - start)
        print(f"  Run {i+1}/{n_runs}: {times_vec[-1]:.4f} s")

    # Verify equivalence
    print("\nVerifying equivalence...")
    max_diff = np.max(np.abs(img_vec - img_ref))
    rel_diff = max_diff / np.max(np.abs(img_ref))
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")

    # Report results
    # Discard first run to avoid warm-up effects
    times_ref = times_ref[1:]
    times_vec = times_vec[1:]
    mean_ref = np.mean(times_ref)
    std_ref = np.std(times_ref)
    mean_vec = np.mean(times_vec)
    std_vec = np.std(times_vec)
    speedup = mean_ref / mean_vec

    print("\n" + "=" * 70)
    print("BEAMFORMING RESULTS")
    print("=" * 70)
    print(f"Reference:   {mean_ref:.4f} ± {std_ref:.4f} s")
    print(f"Vectorized:  {mean_vec:.4f} ± {std_vec:.4f} s")
    print(f"Speedup:     {speedup:.2f}x")
    print("=" * 70)

    return mean_ref, mean_vec, speedup


def benchmark_simulation(n_runs=5):
    """Benchmark forward simulation implementations."""
    print("\n" + "=" * 70)
    print("FORWARD SIMULATION BENCHMARK")
    print("=" * 70)

    # Create test parameters
    P = SimParams()  # Use default parameters from notebook
    scat_pos, scat_amp = sample_scatterers(P, n_scatterers=2000)

    print(f"\nNumber of scatterers: {len(scat_amp)}")
    print(f"Number of angles: {len(P.angles_deg)}")
    print(f"Number of elements: {P.Ne}")

    # Benchmark reference implementation
    print("\nBenchmarking reference implementation...")
    times_ref = []
    for i in range(n_runs):
        start = time.perf_counter()
        signals_ref, _, _, _, _, _ = simulate_forward_channels(P, scat_pos, scat_amp)
        end = time.perf_counter()
        times_ref.append(end - start)
        print(f"  Run {i+1}/{n_runs}: {times_ref[-1]:.4f} s")

    # Benchmark vectorized implementation
    print("\nBenchmarking vectorized implementation...")
    times_vec = []
    for i in range(n_runs):
        start = time.perf_counter()
        signals_vec, _, _, _, _, _ = simulate_forward_channels_vectorized(
            P, scat_pos, scat_amp
        )
        end = time.perf_counter()
        times_vec.append(end - start)
        print(f"  Run {i+1}/{n_runs}: {times_vec[-1]:.4f} s")

    # Verify equivalence
    print("\nVerifying equivalence...")
    max_diff = np.max(np.abs(signals_vec - signals_ref))
    rel_diff = max_diff / np.max(np.abs(signals_ref))
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Max relative difference: {rel_diff:.2e}")

    # Report results
    mean_ref = np.mean(times_ref)
    std_ref = np.std(times_ref)
    mean_vec = np.mean(times_vec)
    std_vec = np.std(times_vec)
    speedup = mean_ref / mean_vec

    print("\n" + "=" * 70)
    print("FORWARD SIMULATION RESULTS")
    print("=" * 70)
    print(f"Reference:   {mean_ref:.4f} ± {std_ref:.4f} s")
    print(f"Vectorized:  {mean_vec:.4f} ± {std_vec:.4f} s")
    print(f"Speedup:     {speedup:.2f}x")
    print("=" * 70)

    return mean_ref, mean_vec, speedup


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("ULTRASOUND PROCESSING PERFORMANCE BENCHMARKS")
    print("=" * 70)

    # Run benchmarks
    beam_ref, beam_vec, beam_speedup = benchmark_beamforming(n_runs=5)
    sim_ref, sim_vec, sim_speedup = benchmark_simulation(n_runs=5)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Beamforming speedup:         {beam_speedup:.2f}x")
    print(f"Forward simulation speedup:  {sim_speedup:.2f}x")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
