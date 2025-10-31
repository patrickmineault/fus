"""
Backward reconstruction v3: Better caching and parallelization

Key insight: Cache aperture weights per depth, not per pixel
"""
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, Literal
import numba


@dataclass
class BeamformConfig:
    c: float = 1540.0
    f0: float = 10e6
    angles_deg: Sequence[float] = (0.0,)
    f_number: Optional[float] = 1.5
    apod_kind: Literal["hann","rect"] = "hann"
    coherent_compound: bool = True


def make_image_grid(
    x_span: Tuple[float, float],
    z_span: Tuple[float, float],
    dx: float, dz: float
):
    x = np.arange(x_span[0], x_span[1] + 1e-12, dx)
    z = np.arange(z_span[0], z_span[1] + 1e-12, dz)
    X, Z = np.meshgrid(x, z)
    return x, z, X, Z


@numba.jit(nopython=True, cache=True)
def compute_apod_weights(
    x_row: np.ndarray,
    zf: float,
    elem_x: np.ndarray,
    f_number: float,
    use_hann: bool
) -> np.ndarray:
    """Compute aperture weights for one row."""
    nx = len(x_row)
    Ne = len(elem_x)
    w = np.zeros((nx, Ne))

    if f_number <= 0:
        # Full aperture
        D_full = elem_x.max() - elem_x.min() + 1e-12
        for n in range(Ne):
            xi = (elem_x[n] - (elem_x.max() + elem_x.min())/2) / D_full
            if abs(xi) <= 0.5:
                if use_hann:
                    u = xi + 0.5
                    w_n = 0.5 * (1.0 - np.cos(2*np.pi*u))
                else:
                    w_n = 1.0
                w[:, n] = w_n
    else:
        # Dynamic aperture
        D = max(zf / f_number, 1e-12)
        for ix in range(nx):
            for n in range(Ne):
                xi = (elem_x[n] - x_row[ix]) / D
                if abs(xi) <= 0.5:
                    if use_hann:
                        u = xi + 0.5
                        w[ix, n] = 0.5 * (1.0 - np.cos(2*np.pi*u))
                    else:
                        w[ix, n] = 1.0

    # Normalize per pixel
    for ix in range(nx):
        s = 0.0
        for n in range(Ne):
            s += w[ix, n]
        if s > 1e-12:
            for n in range(Ne):
                w[ix, n] /= s

    return w


@numba.jit(nopython=True, parallel=True, cache=True)
def beamform_rows_parallel(
    Y_m_real: np.ndarray,
    Y_m_imag: np.ndarray,
    x: np.ndarray,
    z_rows: np.ndarray,
    elem_x: np.ndarray,
    elem_z: np.ndarray,
    sin_th: float,
    cos_th: float,
    c: float,
    f0: float,
    t0: float,
    dt: float,
    K: int,
    f_number: float,
    use_hann: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Beamform multiple rows in parallel."""
    nz = len(z_rows)
    nx = len(x)
    Ne = len(elem_x)

    img_real = np.zeros((nz, nx))
    img_imag = np.zeros((nz, nx))

    # Parallel loop over rows
    for iz in numba.prange(nz):
        zf = z_rows[iz]

        # Compute aperture weights for this row
        w = compute_apod_weights(x, zf, elem_x, f_number, use_hann)

        # Process each lateral pixel
        for ix in range(nx):
            xi = x[ix]
            accum_real = 0.0
            accum_imag = 0.0

            # TX delay
            tau_tx = (xi * sin_th + zf * cos_th) / c

            # Loop over elements
            for n in range(Ne):
                # RX delay
                dx = xi - elem_x[n]
                dz = zf - elem_z[n]
                tau_rx = np.sqrt(dx*dx + dz*dz) / c

                # Total delay
                tau_tot = tau_rx + tau_tx

                # Sample index
                k_idx = int(np.rint((tau_tot - t0) / dt))
                if k_idx < 0:
                    k_idx = 0
                elif k_idx >= K:
                    k_idx = K - 1

                # Phase correction
                phase_arg = 2.0 * np.pi * f0 * tau_tot
                phase_cos = np.cos(phase_arg)
                phase_sin = np.sin(phase_arg)

                # Get sample and apply phase
                sample_real = Y_m_real[n, k_idx]
                sample_imag = Y_m_imag[n, k_idx]

                # Complex multiplication
                weighted_real = sample_real * phase_cos - sample_imag * phase_sin
                weighted_imag = sample_real * phase_sin + sample_imag * phase_cos

                # Apply weight
                accum_real += w[ix, n] * weighted_real
                accum_imag += w[ix, n] * weighted_imag

            img_real[iz, ix] = accum_real
            img_imag[iz, ix] = accum_imag

    return img_real, img_imag


def beamform_plane_wave_das_nn_vec(
    Y: np.ndarray,
    t: np.ndarray,
    elem_pos: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    cfg: BeamformConfig
) -> np.ndarray:
    """Optimized beamforming with parallel row processing."""
    assert Y.ndim == 3, "Y must be (M, Ne, K)"
    M, Ne, K = Y.shape
    nx, nz = x.size, z.size

    t0 = float(t[0])
    dt = float(t[1] - t[0])

    elem_x = elem_pos[:, 0].astype(np.float64)
    elem_z = elem_pos[:, 1].astype(np.float64)

    thetas = np.deg2rad(np.asarray(cfg.angles_deg, dtype=np.float64))

    f_num = cfg.f_number if cfg.f_number is not None else -1.0
    use_hann = cfg.apod_kind == "hann"

    img_complex = np.zeros((nz, nx), dtype=np.complex128)

    for m, th in enumerate(thetas):
        sin_th = np.sin(th)
        cos_th = np.cos(th)

        Y_m_real = np.ascontiguousarray(Y[m].real)
        Y_m_imag = np.ascontiguousarray(Y[m].imag)

        # Process all rows in parallel!
        img_real, img_imag = beamform_rows_parallel(
            Y_m_real, Y_m_imag, x, z, elem_x, elem_z,
            sin_th, cos_th, cfg.c, cfg.f0, t0, dt, K,
            f_num, use_hann
        )

        row_sum = img_real + 1j * img_imag

        if cfg.coherent_compound:
            img_complex += row_sum
        else:
            img_complex += np.abs(row_sum)

    return img_complex
