"""
Reference implementations and vectorized versions of ultrasound processing functions.
Extracted from 02 - Structural Reconstruction.ipynb
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numba import njit, prange
from scipy import fft as sp_fft
from scipy.signal import fftconvolve
from scipy.sparse import csr_matrix

# ========== Reference Implementations ==========


def beamform_das(
    Y,  # (M, Ne, K) complex RF (analytic) channels; M == len(betas)
    times,  # (K,) time stamps [s], uniform
    elem_pos,  # (Ne,) element x-positions [m], z=0
    x,
    z,  # image grids: x (nx,), z (nz,)
    c,  # speed of sound [m/s]
    betas,  # (M,) steering angles [rad], order must match Y's first dim
    omega,  # angular frequency [rad/s]
    alpha_np_per_m=None,  # float or None. If set, apply exp(+alpha * path) TGC (two-way)
):
    """
    Delay-and-sum beamforming for ultrasound image reconstruction.

    Parameters:
    -----------
    Y : ndarray (M, Ne, K)
        Complex RF (analytic) channels for M angles, Ne elements, K time samples
    times : ndarray (K,)
        Time stamps in seconds, uniformly sampled
    elem_pos : ndarray (Ne,)
        Element x-positions in meters at z=0
    x : ndarray (nx,)
        Image x-coordinates in meters
    z : ndarray (nz,)
        Image z-coordinates in meters
    c : float
        Speed of sound in m/s
    betas : ndarray (M,)
        Steering angles in radians
    alpha_np_per_m : float or None
        Attenuation coefficient in Nepers per meter for time-gain compensation

    Returns:
    --------
    img : ndarray (nz, nx)
        Complex beamformed image
    """
    M, Ne, K = Y.shape
    assert M == len(betas), "Y's first dim must match len(betas)."

    t0 = float(times[0])
    dt = float(times[1] - times[0])

    # Precompute lateral offsets per row (nx, Ne), reused for all depths
    dx_row = x[:, None] - elem_pos[None, :]  # (nx, Ne)

    img = np.zeros((z.size, x.size), dtype=np.complex128)

    for m, beta in enumerate(betas):
        sin_beta, cos_beta = np.sin(beta), np.cos(beta)
        Y_m = Y[m]  # (Ne, K)

        for iz, zf in enumerate(z):
            # Two-way delay: τ_tot(x,n) = τ_rx(x,z; n) + τ_tx(x,z; θ)
            tau_rx = np.sqrt(dx_row**2 + zf**2) / c  # (nx, Ne)
            tau_tx_row = (x * sin_beta + zf * cos_beta) / c  # (nx,)
            tau_tot = tau_rx + tau_tx_row[:, None]  # (nx, Ne)

            # Nearest-neighbor sample
            k_idx = np.rint((tau_tot - t0) / dt).astype(np.int64)
            np.clip(k_idx, 0, K - 1, out=k_idx)

            # Gather and sum over elements (uniform weights)
            elem_idx = np.broadcast_to(np.arange(Ne)[None, :], k_idx.shape)
            samples = Y_m[elem_idx, k_idx]  # (nx, Ne)

            # Exponential attenuation compensation (TGC): exp(+alpha * (r_tx + r_rx))
            if alpha_np_per_m is not None and alpha_np_per_m != 0.0:
                path_len = tau_tot * c  # (nx, Ne)
                tgc = np.exp(alpha_np_per_m * path_len)
                samples = samples * tgc

            # Apply phase shift to baseband
            phase_ang = omega * tau_tot  # (nx, Ne)
            samples = samples * np.exp(1j * phase_ang)

            row_sum = samples.mean(axis=1)
            img[iz, :] += row_sum

    return img  # complex image (before envelope/log)


@njit(parallel=True, fastmath=True)
def _build_sparse_matrix_kernel(
    x,
    z,
    elem_pos,
    times,
    betas,
    sin_betas,
    cos_betas,
    c,
    omega,
    t0,
    dt,
    K,
    alpha_np_per_m,
):
    """
    Numba-accelerated kernel to build sparse matrix components.

    Returns row_indices, col_indices, data_real, data_imag arrays.
    """
    M = len(betas)
    Ne = len(elem_pos)
    nx = len(x)
    nz = len(z)

    # Total number of non-zero entries: nz * nx * M * Ne
    nnz = nz * nx * M * Ne

    # Pre-allocate arrays
    row_indices = np.empty(nnz, dtype=np.int64)
    col_indices = np.empty(nnz, dtype=np.int64)
    data_real = np.empty(nnz, dtype=np.float64)
    data_imag = np.empty(nnz, dtype=np.float64)

    # Normalization factor (average over elements)
    norm_factor = 1.0 / Ne
    tgc = 1.0

    # Parallel loop over output pixels
    for idx in prange(nz * nx):
        # Decode flat index to (iz, ix)
        iz = idx // nx
        ix = idx % nx

        zf = z[iz]
        xf = x[ix]

        # Loop over angles and elements
        for m in range(M):
            sin_beta = sin_betas[m]
            cos_beta = cos_betas[m]

            # TX delay for this pixel
            tau_tx = (xf * sin_beta + zf * cos_beta) / c

            for n in range(Ne):
                # RX delay
                dx = xf - elem_pos[n]
                r_rx = np.sqrt(dx * dx + zf * zf)
                tau_rx = r_rx / c
                tau_tot = tau_rx + tau_tx

                # Sample index
                k_idx_float = (tau_tot - t0) / dt
                k_idx = int(np.rint(k_idx_float))
                if k_idx < 0:
                    k_idx = 0
                elif k_idx >= K:
                    k_idx = K - 1

                if alpha_np_per_m is not None and alpha_np_per_m != 0.0:
                    path_len = tau_tot * c
                    tgc = np.exp(alpha_np_per_m * path_len)

                # Phase shift
                phase_ang = omega * tau_tot
                phase_r = np.cos(phase_ang) * norm_factor * tgc
                phase_i = np.sin(phase_ang) * norm_factor * tgc

                # Input index in flattened array
                input_idx = m * Ne * K + n * K + k_idx

                # Calculate position in output arrays
                entry_idx = idx * M * Ne + m * Ne + n

                # Store values
                row_indices[entry_idx] = idx
                col_indices[entry_idx] = input_idx
                data_real[entry_idx] = phase_r
                data_imag[entry_idx] = phase_i

    return row_indices, col_indices, data_real, data_imag


def build_das_sparse_matrix(
    times,  # (K,) time stamps [s], uniform
    elem_pos,  # (Ne,) element x-positions [m], z=0
    x,
    z,  # image grids: x (nx,), z (nz,)
    c,  # speed of sound [m/s]
    betas,  # (M,) steering angles [rad]
    omega,  # angular frequency [rad/s]
    alpha_np_per_m=0.0,  # float; assumed 0 for this optimized version
):
    """
    Build a sparse projection operator X such that the DAS image y = X @ w,
    where w is the flattened (complex) IQ data of shape (M * Ne * K,).

    This version assumes alpha_np_per_m=0 (no TGC compensation).
    Optimized with Numba for fast construction.

    Parameters:
    -----------
    times : ndarray (K,)
        Time stamps in seconds, uniformly sampled
    elem_pos : ndarray (Ne,)
        Element x-positions in meters at z=0
    x : ndarray (nx,)
        Image x-coordinates in meters
    z : ndarray (nz,)
        Image z-coordinates in meters
    c : float
        Speed of sound in m/s
    betas : ndarray (M,)
        Steering angles in radians
    omega : float
        Angular frequency in rad/s

    Returns:
    --------
    X : scipy.sparse.csr_matrix (nz * nx, M * Ne * K)
        Sparse projection operator where each row corresponds to one output pixel
        and encodes which input samples contribute with what phase shifts
    """
    M = len(betas)
    Ne = len(elem_pos)
    K = len(times)
    nx = len(x)
    nz = len(z)

    t0 = float(times[0])
    dt = float(times[1] - times[0])

    # Precompute sin/cos
    sin_betas = np.sin(betas)
    cos_betas = np.cos(betas)

    # Ensure arrays are contiguous
    x = np.ascontiguousarray(x, dtype=np.float64)
    z = np.ascontiguousarray(z, dtype=np.float64)
    elem_pos = np.ascontiguousarray(elem_pos, dtype=np.float64)
    times = np.ascontiguousarray(times, dtype=np.float64)
    betas = np.ascontiguousarray(betas, dtype=np.float64)
    sin_betas = np.ascontiguousarray(sin_betas, dtype=np.float64)
    cos_betas = np.ascontiguousarray(cos_betas, dtype=np.float64)

    # Call Numba kernel
    row_indices, col_indices, data_real, data_imag = _build_sparse_matrix_kernel(
        x,
        z,
        elem_pos,
        times,
        betas,
        sin_betas,
        cos_betas,
        c,
        omega,
        t0,
        dt,
        K,
        alpha_np_per_m,
    )

    # Combine real and imaginary parts
    data_values = data_real + 1j * data_imag

    # Create sparse matrix in CSR format (efficient for matrix-vector products)
    X = csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(nz * nx, M * Ne * K),
        dtype=np.complex128,
    )

    return X


def beamform_das_sparse_with_shape(
    Y,  # (M, Ne, K) complex RF (analytic) channels
    X,  # Precomputed sparse projection operator
    nz,
    nx,
):
    """
    Delay-and-sum beamforming using precomputed sparse projection operator.

    Parameters:
    -----------
    Y : ndarray (M, Ne, K)
        Complex RF (analytic) channels for M angles, Ne elements, K time samples
    X : scipy.sparse matrix (nz * nx, M * Ne * K)
        Sparse projection operator from build_das_sparse_matrix
    nz, nx : int
        Image dimensions

    Returns:
    --------
    img : ndarray (nz, nx)
        Complex beamformed image
    """
    M, Ne, K = Y.shape

    # Flatten input data
    w = Y.reshape(-1)  # (M * Ne * K,)

    # Apply sparse matrix multiplication
    y = X @ w  # (nz * nx,)

    # Reshape to image
    img = y.reshape(nz, nx)

    return img


def hann_burst_envelope(f0: float, cycles: int, fs: float) -> np.ndarray:
    """
    Baseband envelope s(t) using numpy's np.hanning with length ≈ cycles * fs / f0.
    Returns a unit-energy window.
    """
    Nt = max(int(round(cycles * fs / f0)), 1)
    s = np.hanning(Nt).astype(np.float64)
    e = np.sqrt(np.sum(s**2))
    return s / e if e > 0 else s


@dataclass
class SimParams:
    """Simulation parameters for ultrasound forward model."""

    # Medium
    c: float = 1540.0

    # Probe
    Ne: int = 128
    pitch: float = 100e-6
    center_x: float = 0.0

    # TX / sampling
    f_carrier: float = 15e6
    fs: float = 30e6
    cycles: int = 14  # Hann-windowed burst length (baseband envelope)
    t_margin: float = 15e-6

    # Angles (deg)
    angles_deg: Tuple[float, ...] = (-8, -6, -4, -2, 0, 2, 4, 6, 8)

    # Propagation/element model
    use_2d_cylindrical: bool = True  # 2D amplitude ~ 1/sqrt(r); else 3D ~ 1/r
    alpha_db_cm_mhz: float = 0.58  # attenuation model (optional): dB/cm/MHz
    use_attenuation: bool = True

    x_span: Tuple[float, float] = (-9e-3, 9e-3)
    z_span: Tuple[float, float] = (10e-3, 40e-3)


def simulate_forward_channels(P: SimParams, scat_pos: np.ndarray, scat_amp: np.ndarray):
    """
    Simulate ultrasound forward propagation from scatterers to transducer elements.

    Parameters:
    -----------
    P : SimParams
        Simulation parameters
    scat_pos : ndarray (S, 2)
        Scatterer positions [x, z] in meters
    scat_amp : ndarray (S,)
        Scatterer amplitudes (reflectivity)

    Returns:
    --------
    signals : ndarray (M, Ne, K)
        Complex RF signals for M angles, Ne elements, K time samples
    times : ndarray (K,)
        Time stamps in seconds
    elem_pos : ndarray (Ne,)
        Element positions in meters
    betas : ndarray (M,)
        Steering angles in radians
    scat_pos : ndarray (S, 2)
        Scatterer positions (passthrough)
    s_env : ndarray
        Burst envelope
    """
    c = P.c
    elem_pos = make_array_positions(P.Ne, P.pitch, P.center_x)  # (Ne,)
    betas = np.deg2rad(np.array(P.angles_deg, dtype=np.float64))  # (M,)
    s_env = hann_burst_envelope(P.f_carrier, P.cycles, P.fs)
    M, Ne = len(betas), P.Ne

    # Phantom
    S = scat_pos.shape[0]

    # Times (enough to cover max delay + envelope + margin)
    # Worst-case: far corner of FOV for TX + RX to far-most element
    # Max TX delay over angles at FOV far corner:
    x_far, z_far = P.x_span[1], P.z_span[1]
    tx_max = np.max(
        [(x_far * np.sin(beta) + z_far * np.cos(beta)) / c for beta in betas]
    )
    # Max RX delay across elements for far scatterers:
    dx = scat_pos[:, 0][:, None] - elem_pos[None, :]
    r = np.sqrt(dx * dx + scat_pos[:, 1][:, None] ** 2)
    rx_max = r.max() / c
    K = int(np.ceil((tx_max + rx_max + len(s_env) / P.fs + P.t_margin) * P.fs))
    times = np.arange(K) / P.fs

    # Attenuation coefficient (Np/m)
    if P.use_attenuation:
        alpha = (P.alpha_db_cm_mhz * (P.f_carrier / 1e6) * 100.0) / 8.686
    else:
        alpha = 0.0

    # Precompute geometry
    k = 2 * np.pi * P.f_carrier / c
    b = P.pitch / 2  # element half-width for directivity
    # Spreading factor function
    if P.use_2d_cylindrical:
        spread = lambda r_: 1.0 / np.sqrt(np.maximum(r_, 1e-9))
    else:
        spread = lambda r_: 1.0 / np.maximum(r_, 1e-9)

    # Output
    signals = np.zeros((M, Ne, K), dtype=np.complex128)

    # Vector helpers reused in loops
    x_s = scat_pos[:, 0]  # (S,)
    z_s = scat_pos[:, 1]  # (S,)

    for i, beta in enumerate(betas):
        # TX delay to each scatterer (S,)

        r_tx = x_s * np.sin(beta) + z_s * np.cos(beta)
        tau_tx = r_tx / c

        for n in range(Ne):
            # RX distance & delay from scatterers to element n
            r_rx = np.sqrt((x_s - elem_pos[n]) ** 2 + z_s**2)  # (S,)
            tau_rx = r_rx / c
            tau_tot = tau_tx + tau_rx

            # Amplitude: attenuation * spreading * element directivity (receive)
            directivity = np.sinc((k * b * (x_s - elem_pos[n]) / r_rx) / np.pi)
            amp_mag = scat_amp * spread(r_rx) * np.exp(-alpha * (r_tx + r_rx))
            phase = np.exp(-1j * 2 * np.pi * P.f_carrier * tau_tot)
            amps = amp_mag * directivity * phase  # (S,)

            # Fractional-delay spike train (two-tap linear)
            g = np.zeros(K, dtype=np.complex128)
            kf = tau_tot * P.fs
            k0 = np.floor(kf).astype(np.int64)
            frac = kf - k0
            # lower
            valid0 = (k0 >= 0) & (k0 < K)
            np.add.at(g, k0[valid0], amps[valid0] * (1.0 - frac[valid0]))
            # upper
            k1 = k0 + 1
            valid1 = (k1 >= 0) & (k1 < K)
            np.add.at(g, k1[valid1], amps[valid1] * frac[valid1])

            # Convolve with envelope and crop
            y = fftconvolve(g, s_env, mode="full")[:K]
            signals[i, n, :] = y

    return signals, times, elem_pos, betas, scat_pos, s_env


# ========== Numba-Optimized Implementations ==========


@njit(parallel=True, fastmath=True)
def _beamform_das_numba_kernel(
    Y_real,
    Y_imag,  # (M, Ne, K) - split into real and imag parts
    times,
    elem_pos,
    x,
    z,
    c,
    sin_betas,
    cos_betas,  # (M,) - precomputed
    alpha_np_per_m,
    use_tgc,
    omega,
):
    """
    Numba-compiled kernel for beamforming.
    Uses parallel loops over depth for speed while maintaining memory efficiency.
    """
    M, Ne, K = Y_real.shape
    nx = len(x)
    nz = len(z)

    t0 = times[0]
    dt = times[1] - times[0]

    # Output image (real and imag parts)
    img_real = np.zeros((nz, nx), dtype=np.float64)
    img_imag = np.zeros((nz, nx), dtype=np.float64)

    # Precompute lateral offsets (nx, Ne)
    dx_row = np.empty((nx, Ne), dtype=np.float64)
    for ix in range(nx):
        for n in range(Ne):
            dx_row[ix, n] = x[ix] - elem_pos[n]

    # Loop over angles
    for m in range(M):
        sin_beta = sin_betas[m]
        cos_beta = cos_betas[m]

        # Loop over depths in parallel
        for iz in prange(nz):
            zf = z[iz]

            # Temporary arrays for this row
            row_sum_real = 0.0
            row_sum_imag = 0.0

            # Loop over x positions
            for ix in range(nx):
                xf = x[ix]

                # TX delay for this pixel
                tau_tx = (xf * sin_beta + zf * cos_beta) / c

                # Accumulate over elements
                elem_sum_real = 0.0
                elem_sum_imag = 0.0

                for n in range(Ne):
                    # RX delay
                    dx = dx_row[ix, n]
                    tau_rx = np.sqrt(dx * dx + zf * zf) / c
                    tau_tot = tau_rx + tau_tx

                    # Sample index
                    k_idx = int(np.rint((tau_tot - t0) / dt))
                    if k_idx < 0:
                        k_idx = 0
                    elif k_idx >= K:
                        k_idx = K - 1

                    # Get sample
                    sample_real = Y_real[m, n, k_idx]
                    sample_imag = Y_imag[m, n, k_idx]

                    # Apply TGC if needed
                    if use_tgc:
                        path_len = tau_tot * c
                        tgc = np.exp(alpha_np_per_m * path_len)
                        sample_real *= tgc
                        sample_imag *= tgc

                    phase_ang = omega * tau_tot
                    phase_r = np.cos(phase_ang)
                    phase_i = np.sin(phase_ang)

                    elem_sum_real += sample_real * phase_r - sample_imag * phase_i
                    elem_sum_imag += sample_real * phase_i + sample_imag * phase_r

                # Average over elements
                elem_sum_real /= Ne
                elem_sum_imag /= Ne

                # Accumulate to output
                img_real[iz, ix] += elem_sum_real
                img_imag[iz, ix] += elem_sum_imag

    return img_real, img_imag


def beamform_das_vectorized(
    Y, times, elem_pos, x, z, c, betas, omega, alpha_np_per_m=None
):
    """
    Numba-optimized version of beamform_das.
    Much faster than reference implementation with minimal memory overhead.

    Parameters are identical to beamform_das.
    """
    M, Ne, K = Y.shape
    assert M == len(betas), "Y's first dim must match len(betas)."

    # Precompute sin/cos
    sin_betas = np.sin(betas)
    cos_betas = np.cos(betas)

    # Split complex array into real and imaginary parts for Numba
    Y_real = np.ascontiguousarray(Y.real)
    Y_imag = np.ascontiguousarray(Y.imag)

    # Prepare TGC parameters
    if alpha_np_per_m is None or alpha_np_per_m == 0.0:
        use_tgc = False
        alpha_np_per_m_val = 0.0
    else:
        use_tgc = True
        alpha_np_per_m_val = float(alpha_np_per_m)

    # Call Numba kernel
    img_real, img_imag = _beamform_das_numba_kernel(
        Y_real,
        Y_imag,
        times,
        elem_pos,
        x,
        z,
        c,
        sin_betas,
        cos_betas,
        alpha_np_per_m_val,
        use_tgc,
        omega,
    )

    # Combine back to complex
    img = img_real + 1j * img_imag

    return img


@njit(parallel=True, fastmath=True, cache=True)
def _simulate_one_angle_numba(
    sin_beta,
    cos_beta,
    x_s,
    z_s,
    scat_amp,
    tau_rx_all,
    r_rx_all,
    dx_rx_all,  # Precomputed RX geometry
    K,
    fs,
    f_carrier,
    c,
    alpha,
    k,
    b,
    use_2d_cylindrical,
):
    """
    Numba kernel to process one angle efficiently.
    Key optimization: RX geometry is precomputed, only TX changes per angle.
    Parallelizes over elements (Ne).
    """
    Ne = tau_rx_all.shape[1]  # (S, Ne)
    S = len(scat_amp)

    # Output
    g_real = np.zeros((Ne, K), dtype=np.float64)
    g_imag = np.zeros((Ne, K), dtype=np.float64)

    # Constants
    two_pi_fc = 2.0 * np.pi * f_carrier
    kb = k * b

    # Compute TX delays for this angle (same for all elements)
    tau_tx = np.empty(S, dtype=np.float64)
    r_tx = np.empty(S, dtype=np.float64)
    for s in range(S):
        r_tx[s] = x_s[s] * sin_beta + z_s[s] * cos_beta
        tau_tx[s] = r_tx[s] / c

    # Parallel over elements
    for n in prange(Ne):
        # Get precomputed RX geometry for this element
        tau_rx_n = tau_rx_all[:, n]
        r_rx_n = r_rx_all[:, n]
        dx_rx_n = dx_rx_all[:, n]

        # Process all scatterers for this element
        for s in range(S):
            amp = scat_amp[s]

            # Total delay
            tau_tot = tau_tx[s] + tau_rx_n[s]

            # Directivity
            # Note: reference uses np.sinc((kb * dx / r) / np.pi)
            # np.sinc(x) = sin(pi*x)/(pi*x), so np.sinc(arg/pi) = sin(arg)/arg
            arg = kb * dx_rx_n[s] / r_rx_n[s]
            if abs(arg) < 1e-10:
                directivity = 1.0
            else:
                directivity = np.sin(arg) / arg

            # Spreading
            if use_2d_cylindrical:
                spreading = 1.0 / np.sqrt(max(r_rx_n[s], 1e-9))
            else:
                spreading = 1.0 / max(r_rx_n[s], 1e-9)

            # Attenuation
            path_len = r_tx[s] + r_rx_n[s]
            attenuation = np.exp(-alpha * path_len)

            # Combined amplitude
            amp_mag = amp * spreading * directivity * attenuation

            # Phase
            phase_ang = -two_pi_fc * tau_tot
            phase_r = np.cos(phase_ang)
            phase_i = np.sin(phase_ang)

            # Apply cylindrical phase
            total_r = amp_mag * phase_r
            total_i = amp_mag * phase_i

            # Fractional delay
            kf = tau_tot * fs
            k0 = int(np.floor(kf))
            frac = kf - k0
            k1 = k0 + 1

            # Add to spike train
            if 0 <= k0 < K:
                g_real[n, k0] += total_r * (1.0 - frac)
                g_imag[n, k0] += total_i * (1.0 - frac)
            if 0 <= k1 < K:
                g_real[n, k1] += total_r * frac
                g_imag[n, k1] += total_i * frac

    return g_real, g_imag


def simulate_forward_channels_vectorized(
    P: SimParams, scat_pos: np.ndarray, scat_amp: np.ndarray
):
    """
    Numba-optimized version of simulate_forward_channels.
    Key insight from fast_forward.py: Precompute RX geometry once!

    Parameters are identical to simulate_forward_channels.
    """
    c = P.c
    elem_pos = make_array_positions(P.Ne, P.pitch, P.center_x)
    betas = np.deg2rad(np.array(P.angles_deg, dtype=np.float64))
    s_env = hann_burst_envelope(P.f_carrier, P.cycles, P.fs)
    M, Ne = len(betas), P.Ne

    # Scatterer data
    x_s = scat_pos[:, 0].astype(np.float64)
    z_s = scat_pos[:, 1].astype(np.float64)
    scat_amp_f64 = scat_amp.astype(np.float64)
    S = len(x_s)

    # KEY OPTIMIZATION: Precompute RX geometry (doesn't depend on angle!)
    # Shape: (S, Ne) for all scatterers and elements
    dx_rx_all = x_s[:, None] - elem_pos[None, :]  # (S, Ne)
    dz_rx_all = z_s[:, None] - np.zeros(Ne)[None, :]  # (S, Ne)
    r_rx_all = np.sqrt(dx_rx_all**2 + dz_rx_all**2)  # (S, Ne)
    tau_rx_all = r_rx_all / c  # (S, Ne)

    # Compute time array
    x_far, z_far = P.x_span[1], P.z_span[1]
    tx_max = np.max(
        [(x_far * np.sin(beta) + z_far * np.cos(beta)) / c for beta in betas]
    )
    rx_max = tau_rx_all.max()
    K = int(np.ceil((tx_max + rx_max + len(s_env) / P.fs + P.t_margin) * P.fs))
    times = np.arange(K) / P.fs

    # Attenuation
    if P.use_attenuation:
        alpha = (P.alpha_db_cm_mhz * (P.f_carrier / 1e6) * 100.0) / 8.686
    else:
        alpha = 0.0

    k = 2 * np.pi * P.f_carrier / c
    b = P.pitch / 2

    # Output
    signals = np.zeros((M, Ne, K), dtype=np.complex128)

    # Loop over angles (small loop - only ~9 iterations)
    for i, beta in enumerate(betas):
        sin_beta = np.sin(beta)
        cos_beta = np.cos(beta)

        # Generate spike trains for this angle (parallelizes over elements)
        g_real, g_imag = _simulate_one_angle_numba(
            sin_beta,
            cos_beta,
            x_s,
            z_s,
            scat_amp_f64,
            tau_rx_all,
            r_rx_all,
            dx_rx_all,
            K,
            P.fs,
            P.f_carrier,
            c,
            alpha,
            k,
            b,
            P.use_2d_cylindrical,
        )

        g = g_real + 1j * g_imag

        # Vectorized convolution using scipy.fft (faster than numpy.fft)
        conv_len = K + len(s_env) - 1
        G = sp_fft.fft(g, n=conv_len, axis=-1)  # (Ne, conv_len)
        S_env_fft = sp_fft.fft(s_env, n=conv_len)  # (conv_len,)
        Y = G * S_env_fft[None, :]  # (Ne, conv_len)
        y = sp_fft.ifft(Y, axis=-1)[..., :K]  # (Ne, K)

        signals[i, :, :] = y

    return signals, times, elem_pos, betas, scat_pos, s_env


def make_array_positions(Ne: int, pitch: float, center_x: float) -> np.ndarray:
    idx = np.arange(Ne) - (Ne - 1) / 2
    return center_x + idx * pitch  # (Ne,) x-positions at z=0


def make_image_grid(x_span, z_span, dx, dz):
    x = np.arange(x_span[0], x_span[1] + 1e-12, dx)
    z = np.arange(z_span[0], z_span[1] + 1e-12, dz)
    X, Z = np.meshgrid(x, z)  # (nz, nx)
    return x, z, X, Z
