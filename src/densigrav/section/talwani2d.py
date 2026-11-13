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
    vertices define a simple polygon; either orientation is accepted.

    The polygon may lie anywhere relative to the observation points: below them,
    above them (negative gz), or straddling their height (e.g. a body that reaches
    the ground surface next to a station). An observation point inside the body is
    also handled; a point exactly on the boundary gets the finite limiting value.

    Each edge contributes (line-integral form of Talwani et al., 1959; see also
    Won and Bevis, 1987)

        c * (dx * dtheta + dz * ln(r2 / r1)),   c = (x1 z2 - x2 z1) / (dx^2 + dz^2)

    where (x, z) are vertex coordinates relative to the observation point,
    theta = atan2(z, x), and dtheta = theta1 - theta2 is the angle swept by the
    edge as seen from the observation point, wrapped into (-pi, pi]. Wrapping the
    swept angle (rather than each theta separately) is what keeps the formula
    valid when an edge crosses the height of the observation point. Horizontal and
    vertical edges need no special treatment.
    """
    x_obs = np.asarray(x_obs_m, dtype=float).ravel()
    z_obs = np.asarray(z_obs_m, dtype=float).ravel()
    if x_obs.shape != z_obs.shape:
        raise ValueError("x_obs and z_obs must have the same shape.")

    verts = _close_polygon(np.asarray(vertices_xz_m, dtype=float))
    if _signed_area_xz(verts) < 0:
        # normalise the orientation so that a positive density contrast below the
        # observation point always gives positive gz
        verts = verts[::-1].copy()

    xv = verts[:, 0]
    zv = verts[:, 1]
    acc = np.zeros_like(x_obs, dtype=float)

    for i in range(len(verts) - 1):
        x1 = xv[i] - x_obs
        z1 = zv[i] - z_obs
        x2 = xv[i + 1] - x_obs
        z2 = zv[i + 1] - z_obs
        dx = float(xv[i + 1] - xv[i])
        dz = float(zv[i + 1] - zv[i])
        len2 = dx * dx + dz * dz
        if len2 == 0.0:
            continue  # repeated vertex

        r1 = np.hypot(x1, z1)
        r2 = np.hypot(x2, z2)
        cross = x1 * z2 - x2 * z1
        # Observation point on the line through the edge (including at a vertex):
        # c -> 0 linearly while the log term diverges only logarithmically, so the
        # contribution tends to zero.
        on_line = np.abs(cross) <= 1e-9 * np.sqrt(len2) * np.maximum(np.maximum(r1, r2), 1.0)
        safe = ~on_line & (r1 > 0.0) & (r2 > 0.0)

        dth = np.arctan2(z1, x1) - np.arctan2(z2, x2)
        dth = np.where(dth > np.pi, dth - 2.0 * np.pi, dth)
        dth = np.where(dth <= -np.pi, dth + 2.0 * np.pi, dth)

        log_term = np.zeros_like(r1)
        log_term[safe] = np.log(r2[safe] / r1[safe])
        contrib = (cross / len2) * (dx * dth + dz * log_term)
        acc += np.where(safe, contrib, 0.0)

    return 2.0 * G_SI * float(density_contrast_kgm3) * acc * SI2MGAL
