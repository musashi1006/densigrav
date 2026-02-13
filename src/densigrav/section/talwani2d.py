from __future__ import annotations

import numpy as np

G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
SI2MGAL = 1e5  # (m/s^2) -> mGal


def _close_polygon(verts: np.ndarray) -> np.ndarray:
    if verts.shape[0] < 3:
        raise ValueError("Polygon needs >= 3 vertices.")
    if not np.allclose(verts[0], verts[-1]):
        return np.vstack([verts, verts[0]])
    return verts


def _signed_area_xz(verts: np.ndarray) -> float:
    # Shoelace on (x, z)
    x = verts[:, 0]
    z = verts[:, 1]
    return 0.5 * float(np.sum(x[:-1] * z[1:] - x[1:] * z[:-1]))


def talwani_gz_polygon(
    x_obs_m: np.ndarray,
    z_obs_m: np.ndarray,
    vertices_xz_m: np.ndarray,
    density_contrast_kgm3: float,
) -> np.ndarray:
    """
    2D Talwani (infinite strike length) vertical component gz in mGal.
    Coordinates: x to the right, z positive downward.
    vertices must define a closed polygon; orientation affects sign.
    """
    x_obs = np.asarray(x_obs_m, dtype=float).ravel()
    z_obs = np.asarray(z_obs_m, dtype=float).ravel()
    if x_obs.shape != z_obs.shape:
        raise ValueError("x_obs and z_obs must have the same shape.")

    verts = _close_polygon(np.asarray(vertices_xz_m, dtype=float))
    area = _signed_area_xz(verts)
    if area < 0:
        # In z-down coords, clockwise is typically negative area depending on convention.
        # To be user-friendly, auto-fix by reversing if needed.
        verts = verts[::-1].copy()

    xv = verts[:, 0]
    zv = verts[:, 1]

    # Vectorized over observation points
    gz = np.zeros_like(x_obs, dtype=float)

    eps = 1e-10
    rho = float(density_contrast_kgm3)

    for i in range(len(verts) - 1):
        x1 = xv[i] - x_obs
        z1 = zv[i] - z_obs
        x2 = xv[i + 1] - x_obs
        z2 = zv[i + 1] - z_obs

        # Avoid singularities
        x1 = np.where(np.abs(x1) < eps, np.sign(x1 + eps) * eps, x1)
        x2 = np.where(np.abs(x2) < eps, np.sign(x2 + eps) * eps, x2)
        z1 = np.where(np.abs(z1) < eps, np.sign(z1 + eps) * eps, z1)
        z2 = np.where(np.abs(z2) < eps, np.sign(z2 + eps) * eps, z2)

        # Parameter from Talwani derivation (stable implementation style)

        dz = z1 - z2

        # Skip (nearly) horizontal edges to avoid division by zero / numerical blow-ups.
        # This stabilizes the computation; their contribution approaches 0 in the limit.
        mask = np.abs(dz) > 1e-12
        if not np.any(mask):
            continue

        a = x2 + z2 * (x2 - x1) / (z1 - z2)

        phi = np.zeros_like(z1)
        phi[mask] = np.arctan2((z2[mask] - z1[mask]), (x2[mask] - x1[mask]))

        a = np.zeros_like(z1)
        a[mask] = x2[mask] + z2[mask] * (x2[mask] - x1[mask]) / dz[mask]

        th1 = np.zeros_like(z1)
        th2 = np.zeros_like(z1)
        th1[mask] = np.arctan2(z1[mask], x1[mask])
        th2[mask] = np.arctan2(z2[mask], x2[mask])

        th1 = np.where(th1 < 0, th1 + np.pi, th1)
        th2 = np.where(th2 < 0, th2 + np.pi, th2)

        cos1 = np.zeros_like(z1)
        cos2 = np.zeros_like(z1)
        tan1 = np.zeros_like(z1)
        tan2 = np.zeros_like(z1)
        tanp = np.zeros_like(z1)

        cos1[mask] = np.cos(th1[mask])
        cos2[mask] = np.cos(th2[mask])
        tan1[mask] = np.tan(th1[mask])
        tan2[mask] = np.tan(th2[mask])
        tanp[mask] = np.tan(phi[mask])

        num = np.zeros_like(z1)
        den = np.zeros_like(z1)
        num[mask] = cos1[mask] * (tan1[mask] - tanp[mask])
        den[mask] = cos2[mask] * (tan2[mask] - tanp[mask])

        ratio = np.ones_like(z1)
        ratio[mask] = np.abs(num[mask] / den[mask])
        ratio = np.where(ratio < eps, eps, ratio)

        contrib = np.zeros_like(z1)
        contrib[mask] = (
            a[mask]
            * np.sin(phi[mask])
            * np.cos(phi[mask])
            * ((th1[mask] - th2[mask]) + tanp[mask] * np.log(ratio[mask]))
        )

        contrib = np.where(np.isfinite(contrib), contrib, 0.0)
        contrib = np.where(np.abs(th1 - th2) < 1e-12, 0.0, contrib)
        gz += contrib

    gz = 2.0 * G_SI * rho * gz * SI2MGAL
    return gz
