"""
Forward simulation v3: Better numba parallelization

Key insight: Process multiple elements in parallel, not just inner loops
"""
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Tuple
from scipy.signal import fftconvolve
import numba


@dataclass
class SimParams:
    # Medium
    c: float = 1540.0
    # Probe geometry
    Ne: int = 128
    pitch: float = 0.0001
    center_x: float = 0.0
    # Transmit
    f0: float = 15e6
    cycles: int = 14
    pulse_window: str = "hann"
    # Angles (degrees)
    angles_deg: Tuple[float, ...] = (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0)
    # Sampling
    fs: float = 50e6
    t_margin: float = 15e-6
    # Field of view / phantom
    x_span: Tuple[float, float] = (-0.009, 0.009)
    z_span: Tuple[float, float] = (0.005, 0.035)
    # Scatterers
    n_bg: int = 8000
    refl_bg: float = 1.0
    refl_inclusion: float = 0.0
    # Inclusion grid
    n_rows: int = 3
    n_cols: int = 3
    min_radius: float = 0.0008
    max_radius: float = 0.003
    grid_margin: float = 0.002
    # Amplitude distance law
    use_range_gain: bool = True


def hann_burst_envelope(f0: float, cycles: int, fs: float) -> Tuple[NDArray, NDArray]:
    T = cycles / f0
    Nt = int(np.ceil(T * fs)) + 1
    t = np.arange(Nt) / fs
    w = 0.5 * (1 - np.cos(2*np.pi * np.clip(t, 0, T)/T))
    w[t > T] = 0.0
    s = w.astype(np.float64)
    e = np.sqrt(np.sum(s**2))
    if e > 0:
        s = s / e
    return s, t


def make_array_positions(Ne: int, pitch: float, center_x: float) -> NDArray:
    idx = np.arange(Ne) - (Ne - 1) / 2.0
    x = center_x + idx * pitch
    return np.column_stack([x, np.zeros_like(x)])


def inclusion_centers_and_radii(P: SimParams) -> Tuple[NDArray, NDArray]:
    x0, x1 = P.x_span
    z0, z1 = P.z_span
    gx0 = x0 + P.grid_margin
    gx1 = x1 - P.grid_margin
    gz0 = z0 + P.grid_margin
    gz1 = z1 - P.grid_margin
    xs = np.linspace(gx0, gx1, P.n_cols)
    zs = np.linspace(gz0, gz1, P.n_rows)
    Xc, Zc = np.meshgrid(xs, zs)
    centers = np.column_stack([Xc.ravel(), Zc.ravel()])
    radii = np.linspace(P.min_radius, P.max_radius, P.n_cols)
    Radii = np.tile(radii, P.n_rows)
    return centers, Radii


def sample_scatterers(P: SimParams, seed: int = 0) -> Tuple[NDArray, NDArray]:
    rng = np.random.default_rng(seed)
    x0, x1 = P.x_span
    z0, z1 = P.z_span
    xs = rng.uniform(x0, x1, P.n_bg)
    zs = rng.uniform(z0, z1, P.n_bg)
    scat = np.column_stack([xs, zs]).astype(np.float64)
    amp = np.full(P.n_bg, P.refl_bg, dtype=np.float64)
    centers, Radii = inclusion_centers_and_radii(P)
    for (cx, cz), R in zip(centers, Radii):
        r2 = (scat[:,0]-cx)**2 + (scat[:,1]-cz)**2
        inside = r2 <= R**2
        amp[inside] = P.refl_inclusion
    return scat, amp


def plane_wave_tx_delay(points: NDArray, theta: float, c: float) -> NDArray:
    x = points[:,0]; z = points[:,1]
    return (x*np.sin(theta) + z*np.cos(theta)) / c


def rx_delay(points: NDArray, elem_pos: NDArray, c: float) -> NDArray:
    dx = points[:,0][:,None] - elem_pos[:,0][None,:]
    dz = points[:,1][:,None] - elem_pos[:,1][None,:]
    r = np.sqrt(dx*dx + dz*dz)
    return r / c, r


@numba.jit(nopython=True, parallel=True, cache=True)
def compute_all_elements_parallel(
    tau_tx: NDArray, rx_delays_s: NDArray, scat_amp: NDArray,
    rdist: NDArray, scat_x: NDArray, elem_x: NDArray,
    f0: float, k: float, pitch: float, fs: float, K: int, Ne: int
) -> Tuple[NDArray, NDArray]:
    """Compute spike trains for ALL elements in parallel using prange."""
    S = len(tau_tx)

    # Output arrays (real and imag parts)
    g_real = np.zeros((Ne, K), dtype=np.float64)
    g_imag = np.zeros((Ne, K), dtype=np.float64)

    # Parallel loop over elements
    for n in numba.prange(Ne):
        tau_rx_n = rx_delays_s[:, n]
        rdist_n = rdist[:, n]
        elem_x_n = elem_x[n]

        # Compute for this element
        tau_tot = tau_tx + tau_rx_n
        rngain = 1.0 / np.maximum(rdist_n, 1e-6)
        amps_mag = scat_amp * rngain

        # Phase
        phase_arg = -2.0 * np.pi * f0 * tau_tot

        # Sensitivity
        for i in range(S):
            arg = k * pitch * (scat_x[i] - elem_x_n) / rdist_n[i]
            if abs(arg) < 1e-10:
                sensitivity_i = 1.0
            else:
                sensitivity_i = np.sin(np.pi * arg) / (np.pi * arg)

            # Complex amplitude
            amp_real = amps_mag[i] * sensitivity_i * np.cos(phase_arg[i])
            amp_imag = amps_mag[i] * sensitivity_i * np.sin(phase_arg[i])

            # Build spike train with linear interpolation
            k_float = tau_tot[i] * fs
            k0 = int(np.floor(k_float))
            frac = k_float - k0

            if 0 <= k0 < K:
                g_real[n, k0] += amp_real * (1.0 - frac)
                g_imag[n, k0] += amp_imag * (1.0 - frac)

            k1 = k0 + 1
            if 0 <= k1 < K:
                g_real[n, k1] += amp_real * frac
                g_imag[n, k1] += amp_imag * frac

    return g_real, g_imag


def simulate_forward_channels(P: SimParams):
    """Main driver with parallel element processing."""
    elem_pos = make_array_positions(P.Ne, P.pitch, P.center_x)
    scat_pos, scat_amp = sample_scatterers(P)
    s_env, t_env = hann_burst_envelope(P.f0, P.cycles, P.fs)

    thetas = np.deg2rad(np.array(P.angles_deg, dtype=np.float64))
    M = len(thetas)
    Ne = P.Ne

    tx_max = np.max([plane_wave_tx_delay(
        np.array([[P.x_span[1], P.z_span[1]]]), th, P.c) for th in thetas])
    rx_delays_s, rdist = rx_delay(scat_pos, elem_pos, P.c)
    max_rx = np.max(rx_delays_s)
    t_end = tx_max + max_rx + (len(t_env)/P.fs) + P.t_margin
    dt = 1.0 / P.fs
    K = int(np.floor(t_end / dt)) + 2
    t = np.arange(K) * dt

    Y = np.zeros((M, Ne, K), dtype=np.complex128)
    k = 2 * np.pi * P.f0 / P.c

    elem_x = elem_pos[:, 0]
    scat_x = scat_pos[:, 0]

    # Loop over angles
    for m, th in enumerate(thetas):
        tau_tx = plane_wave_tx_delay(scat_pos, th, P.c)

        # Compute all elements in parallel!
        g_real, g_imag = compute_all_elements_parallel(
            tau_tx, rx_delays_s, scat_amp, rdist, scat_x, elem_x,
            P.f0, k, P.pitch, P.fs, K, Ne
        )

        g = g_real + 1j * g_imag

        # Convolve each element with envelope
        for n in range(Ne):
            y = fftconvolve(g[n], s_env, mode='full')
            Y[m, n, :] = y[:K]

    return Y, t, elem_pos, scat_pos, scat_amp, s_env, t_env
