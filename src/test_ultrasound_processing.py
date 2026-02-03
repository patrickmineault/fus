"""
Unit tests to verify equivalence of vectorized implementations.
"""

import numpy as np
import pytest

from ultrasound_processing import (
    SimParams,
    beamform_das,
    beamform_das_vectorized,
    build_das_sparse_matrix,
    beamform_das_sparse_with_shape,
    make_array_positions,
    simulate_forward_channels,
    simulate_forward_channels_vectorized,
)


def sample_scatterers(P: SimParams, n_scatterers=100, seed=42):
    """Helper function to sample random scatterers for testing."""
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


class TestBeamformDAS:
    """Tests for beamform_das and its vectorized version."""

    @pytest.fixture
    def test_data(self):
        """Create test data for beamforming tests."""
        # Use smaller parameters for faster testing
        P = SimParams(
            Ne=32,  # Fewer elements
            angles_deg=(-4, 0, 4),  # Fewer angles
            x_span=(-5e-3, 5e-3),  # Smaller FOV
            z_span=(15e-3, 25e-3),
        )

        # Generate test scatterers
        scat_pos, scat_amp = sample_scatterers(P, n_scatterers=50)

        # Simulate forward channels
        signals, times, elem_pos, betas, _, _ = simulate_forward_channels(
            P, scat_pos, scat_amp
        )

        # Create image grid
        x, z = make_image_grid(P.x_span, P.z_span, 0.2e-3, 0.2e-3)

        # Compute attenuation coefficient
        if P.use_attenuation:
            alpha = (P.alpha_db_cm_mhz * (P.f_carrier / 1e6) * 100.0) / 8.686
        else:
            alpha = None

        return {
            "signals": signals,
            "times": times,
            "elem_pos": elem_pos,
            "x": x,
            "z": z,
            "c": P.c,
            "betas": betas,
            "omega": 2 * np.pi * P.f_carrier,
            "alpha": alpha,
        }

    def test_beamform_equivalence_no_tgc(self, test_data):
        """Test that vectorized beamforming matches reference implementation without TGC."""
        # Run reference implementation
        img_ref = beamform_das(
            test_data["signals"],
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
            alpha_np_per_m=None,
        )

        # Run vectorized implementation
        img_vec = beamform_das_vectorized(
            test_data["signals"],
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
            alpha_np_per_m=None,
        )

        # Compare
        np.testing.assert_allclose(img_vec, img_ref, rtol=1e-10, atol=1e-12)

    def test_beamform_equivalence_with_tgc(self, test_data):
        """Test that vectorized beamforming matches reference implementation with TGC."""
        # Run reference implementation
        img_ref = beamform_das(
            test_data["signals"],
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
            alpha_np_per_m=test_data["alpha"],
        )

        # Run vectorized implementation
        img_vec = beamform_das_vectorized(
            test_data["signals"],
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
            alpha_np_per_m=test_data["alpha"],
        )

        # Compare
        np.testing.assert_allclose(img_vec, img_ref, rtol=1e-10, atol=1e-12)

    def test_beamform_output_shape(self, test_data):
        """Test that output shape is correct."""
        img = beamform_das_vectorized(
            test_data["signals"],
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
        )

        expected_shape = (len(test_data["z"]), len(test_data["x"]))
        assert img.shape == expected_shape
        assert img.dtype == np.complex128


class TestSimulateForwardChannels:
    """Tests for simulate_forward_channels and its vectorized version."""

    @pytest.fixture
    def test_params(self):
        """Create test parameters."""
        # Use smaller parameters for faster testing
        P = SimParams(
            Ne=32,  # Fewer elements
            angles_deg=(-4, 0, 4),  # Fewer angles
            x_span=(-5e-3, 5e-3),  # Smaller FOV
            z_span=(15e-3, 25e-3),
            cycles=10,  # Fewer cycles
        )
        scat_pos, scat_amp = sample_scatterers(P, n_scatterers=50)
        return P, scat_pos, scat_amp

    def test_simulate_equivalence(self, test_params):
        """Test that vectorized simulation matches reference implementation."""
        P, scat_pos, scat_amp = test_params

        # Run reference implementation
        signals_ref, times_ref, elem_pos_ref, betas_ref, _, s_env_ref = (
            simulate_forward_channels(P, scat_pos, scat_amp)
        )

        # Run vectorized implementation
        signals_vec, times_vec, elem_pos_vec, betas_vec, _, s_env_vec = (
            simulate_forward_channels_vectorized(P, scat_pos, scat_amp)
        )

        # Compare signals (main output)
        np.testing.assert_allclose(signals_vec, signals_ref, rtol=1e-10, atol=1e-12)

        # Compare metadata
        np.testing.assert_array_equal(times_vec, times_ref)
        np.testing.assert_array_equal(elem_pos_vec, elem_pos_ref)
        np.testing.assert_array_equal(betas_vec, betas_ref)
        np.testing.assert_array_equal(s_env_vec, s_env_ref)

    def test_simulate_output_shape(self, test_params):
        """Test that output shapes are correct."""
        P, scat_pos, scat_amp = test_params

        signals, times, elem_pos, betas, _, s_env = (
            simulate_forward_channels_vectorized(P, scat_pos, scat_amp)
        )

        M = len(P.angles_deg)
        Ne = P.Ne

        assert signals.shape[0] == M
        assert signals.shape[1] == Ne
        assert signals.dtype == np.complex128
        assert len(elem_pos) == Ne
        assert len(betas) == M

    def test_simulate_single_scatterer(self):
        """Test with a single scatterer at known position."""
        P = SimParams(
            Ne=16,
            angles_deg=(0,),
            x_span=(-5e-3, 5e-3),
            z_span=(15e-3, 25e-3),
        )

        # Single scatterer at center
        scat_pos = np.array([[0.0, 0.02]])
        scat_amp = np.array([1.0])

        # Run both implementations
        signals_ref, _, _, _, _, _ = simulate_forward_channels(P, scat_pos, scat_amp)
        signals_vec, _, _, _, _, _ = simulate_forward_channels_vectorized(
            P, scat_pos, scat_amp
        )

        # Compare
        np.testing.assert_allclose(signals_vec, signals_ref, rtol=1e-10, atol=1e-12)


class TestEndToEnd:
    """End-to-end tests combining simulation and beamforming."""

    def test_full_pipeline(self):
        """Test full pipeline: simulate -> beamform."""
        # Small test case
        P = SimParams(
            Ne=32,
            angles_deg=(-4, 0, 4),
            x_span=(-5e-3, 5e-3),
            z_span=(15e-3, 25e-3),
        )

        scat_pos, scat_amp = sample_scatterers(P, n_scatterers=30)

        # Reference pipeline
        signals_ref, times, elem_pos, betas, _, _ = simulate_forward_channels(
            P, scat_pos, scat_amp
        )
        x, z = make_image_grid(P.x_span, P.z_span, 0.2e-3, 0.2e-3)
        img_ref = beamform_das(
            signals_ref, times, elem_pos, x, z, P.c, betas, 2 * np.pi * P.f_carrier
        )

        # Vectorized pipeline
        signals_vec, _, _, _, _, _ = simulate_forward_channels_vectorized(
            P, scat_pos, scat_amp
        )
        img_vec = beamform_das_vectorized(
            signals_vec, times, elem_pos, x, z, P.c, betas, 2 * np.pi * P.f_carrier
        )

        # Compare final images
        np.testing.assert_allclose(img_vec, img_ref, rtol=1e-10, atol=1e-12)

        # Verify image has reasonable properties
        assert np.all(np.isfinite(img_ref))
        assert np.all(np.isfinite(img_vec))
        assert np.max(np.abs(img_ref)) > 0  # Not all zeros


class TestSparseMatrixDAS:
    """Tests for sparse matrix DAS implementation."""

    @pytest.fixture
    def test_data(self):
        """Create test data for sparse matrix beamforming tests."""
        # Use smaller parameters for faster testing
        P = SimParams(
            Ne=32,  # Fewer elements
            angles_deg=(-4, 0, 4),  # Fewer angles
            x_span=(-5e-3, 5e-3),  # Smaller FOV
            z_span=(15e-3, 25e-3),
        )

        # Generate test scatterers
        scat_pos, scat_amp = sample_scatterers(P, n_scatterers=50)

        # Simulate forward channels
        signals, times, elem_pos, betas, _, _ = simulate_forward_channels(
            P, scat_pos, scat_amp
        )

        # Create image grid
        x, z = make_image_grid(P.x_span, P.z_span, 0.2e-3, 0.2e-3)

        return {
            "signals": signals,
            "times": times,
            "elem_pos": elem_pos,
            "x": x,
            "z": z,
            "c": P.c,
            "betas": betas,
            "omega": 2 * np.pi * P.f_carrier,
            "P": P,
        }

    def test_sparse_matrix_equivalence_to_reference(self, test_data):
        """Test that sparse matrix beamforming matches reference implementation."""
        # Run reference implementation (no TGC, since sparse doesn't support it)
        img_ref = beamform_das(
            test_data["signals"],
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
            alpha_np_per_m=None,
        )

        # Build sparse matrix
        X = build_das_sparse_matrix(
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
        )

        # Run sparse matrix implementation
        img_sparse = beamform_das_sparse_with_shape(
            test_data["signals"], X, len(test_data["z"]), len(test_data["x"])
        )

        # Compare
        np.testing.assert_allclose(img_sparse, img_ref, rtol=1e-8, atol=1e-10)

    def test_sparse_matrix_equivalence_to_vectorized(self, test_data):
        """Test that sparse matrix beamforming matches vectorized implementation."""
        # Run vectorized implementation (no TGC)
        img_vec = beamform_das_vectorized(
            test_data["signals"],
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
            alpha_np_per_m=None,
        )

        # Build sparse matrix
        X = build_das_sparse_matrix(
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
        )

        # Run sparse matrix implementation
        img_sparse = beamform_das_sparse_with_shape(
            test_data["signals"], X, len(test_data["z"]), len(test_data["x"])
        )

        # Compare
        np.testing.assert_allclose(img_sparse, img_vec, rtol=1e-8, atol=1e-10)

    def test_sparse_matrix_shape(self, test_data):
        """Test that sparse matrix has correct shape."""
        X = build_das_sparse_matrix(
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
        )

        M = len(test_data["betas"])
        Ne = len(test_data["elem_pos"])
        K = len(test_data["times"])
        nx = len(test_data["x"])
        nz = len(test_data["z"])

        expected_shape = (nz * nx, M * Ne * K)
        assert X.shape == expected_shape
        assert X.dtype == np.complex128

    def test_sparse_matrix_nnz(self, test_data):
        """Test that sparse matrix has expected number of non-zeros."""
        X = build_das_sparse_matrix(
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
        )

        M = len(test_data["betas"])
        Ne = len(test_data["elem_pos"])
        nx = len(test_data["x"])
        nz = len(test_data["z"])

        # Each pixel should have exactly M * Ne entries
        expected_nnz = nz * nx * M * Ne
        assert X.nnz == expected_nnz

    def test_sparse_matrix_output_shape(self, test_data):
        """Test that output image has correct shape."""
        X = build_das_sparse_matrix(
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
        )

        img = beamform_das_sparse_with_shape(
            test_data["signals"], X, len(test_data["z"]), len(test_data["x"])
        )

        expected_shape = (len(test_data["z"]), len(test_data["x"]))
        assert img.shape == expected_shape
        assert img.dtype == np.complex128

    def test_sparse_matrix_linearity(self, test_data):
        """Test that sparse matrix multiplication is linear."""
        X = build_das_sparse_matrix(
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
        )

        # Test linearity: X @ (a * w) = a * (X @ w)
        a = 2.5 + 1.3j
        Y1 = test_data["signals"]
        Y2 = a * test_data["signals"]

        img1 = beamform_das_sparse_with_shape(
            Y1, X, len(test_data["z"]), len(test_data["x"])
        )
        img2 = beamform_das_sparse_with_shape(
            Y2, X, len(test_data["z"]), len(test_data["x"])
        )

        np.testing.assert_allclose(img2, a * img1, rtol=1e-10, atol=1e-12)

    def test_sparse_matrix_additivity(self, test_data):
        """Test that sparse matrix multiplication is additive."""
        X = build_das_sparse_matrix(
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
        )

        # Create two different signals with the same shape
        Y1 = test_data["signals"]
        # Create a modified version of the same signal (scaled and with added noise)
        np.random.seed(123)
        noise = (np.random.randn(*Y1.shape) + 1j * np.random.randn(*Y1.shape)) * 0.1
        Y2 = Y1 * 0.5 + noise

        # Test additivity: X @ (w1 + w2) = X @ w1 + X @ w2
        Y_sum = Y1 + Y2

        img1 = beamform_das_sparse_with_shape(
            Y1, X, len(test_data["z"]), len(test_data["x"])
        )
        img2 = beamform_das_sparse_with_shape(
            Y2, X, len(test_data["z"]), len(test_data["x"])
        )
        img_sum = beamform_das_sparse_with_shape(
            Y_sum, X, len(test_data["z"]), len(test_data["x"])
        )

        np.testing.assert_allclose(img_sum, img1 + img2, rtol=1e-10, atol=1e-12)

    def test_sparse_matrix_single_angle(self):
        """Test sparse matrix with single angle."""
        P = SimParams(
            Ne=16,
            angles_deg=(0,),  # Single angle
            x_span=(-3e-3, 3e-3),
            z_span=(15e-3, 20e-3),
        )

        scat_pos, scat_amp = sample_scatterers(P, n_scatterers=20)
        signals, times, elem_pos, betas, _, _ = simulate_forward_channels(
            P, scat_pos, scat_amp
        )
        x, z = make_image_grid(P.x_span, P.z_span, 0.3e-3, 0.3e-3)

        # Reference
        img_ref = beamform_das(
            signals, times, elem_pos, x, z, P.c, betas, 2 * np.pi * P.f_carrier
        )

        # Sparse matrix
        X = build_das_sparse_matrix(times, elem_pos, x, z, P.c, betas, 2 * np.pi * P.f_carrier)
        img_sparse = beamform_das_sparse_with_shape(signals, X, len(z), len(x))

        np.testing.assert_allclose(img_sparse, img_ref, rtol=1e-8, atol=1e-10)

    def test_sparse_matrix_single_element(self):
        """Test sparse matrix with single element (edge case)."""
        P = SimParams(
            Ne=1,  # Single element
            angles_deg=(0,),
            x_span=(-1e-3, 1e-3),
            z_span=(18e-3, 22e-3),
        )

        scat_pos, scat_amp = sample_scatterers(P, n_scatterers=10)
        signals, times, elem_pos, betas, _, _ = simulate_forward_channels(
            P, scat_pos, scat_amp
        )
        x, z = make_image_grid(P.x_span, P.z_span, 0.5e-3, 0.5e-3)

        # Reference
        img_ref = beamform_das(
            signals, times, elem_pos, x, z, P.c, betas, 2 * np.pi * P.f_carrier
        )

        # Sparse matrix
        X = build_das_sparse_matrix(times, elem_pos, x, z, P.c, betas, 2 * np.pi * P.f_carrier)
        img_sparse = beamform_das_sparse_with_shape(signals, X, len(z), len(x))

        np.testing.assert_allclose(img_sparse, img_ref, rtol=1e-8, atol=1e-10)

    def test_sparse_matrix_reusability(self, test_data):
        """Test that sparse matrix can be reused for different signals with same shape."""
        # Build matrix once
        X = build_das_sparse_matrix(
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
        )

        # Process multiple different signal sets with the same matrix
        # Note: Signals must have same shape (M, Ne, K) as the matrix expects
        np.random.seed(100)
        for i in range(3):
            # Create synthetic signals with the same shape as the original
            M, Ne, K = test_data["signals"].shape
            # Random complex signals
            signals = (
                np.random.randn(M, Ne, K) + 1j * np.random.randn(M, Ne, K)
            ) * 0.01

            # Pad to ensure same K if needed
            if signals.shape[-1] != K:
                signals = np.pad(
                    signals,
                    ((0, 0), (0, 0), (0, K - signals.shape[-1])),
                    mode="constant",
                )

            # Reference
            img_ref = beamform_das(
                signals,
                test_data["times"],
                test_data["elem_pos"],
                test_data["x"],
                test_data["z"],
                test_data["c"],
                test_data["betas"],
                test_data["omega"],
            )

            # Sparse (reusing same matrix)
            img_sparse = beamform_das_sparse_with_shape(
                signals, X, len(test_data["z"]), len(test_data["x"])
            )

            np.testing.assert_allclose(img_sparse, img_ref, rtol=1e-8, atol=1e-10)

    def test_sparse_matrix_no_nans_or_infs(self, test_data):
        """Test that sparse matrix contains no NaN or Inf values."""
        X = build_das_sparse_matrix(
            test_data["times"],
            test_data["elem_pos"],
            test_data["x"],
            test_data["z"],
            test_data["c"],
            test_data["betas"],
            test_data["omega"],
        )

        # Check matrix data
        assert np.all(np.isfinite(X.data))

        # Check output
        img = beamform_das_sparse_with_shape(
            test_data["signals"], X, len(test_data["z"]), len(test_data["x"])
        )
        assert np.all(np.isfinite(img))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
