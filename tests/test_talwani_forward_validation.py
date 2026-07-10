"""Validation of the 2-D Talwani forward against independent ground truths.

Three independent checks, deliberately covering edge orientations:

1. Buried horizontal cylinder vs the analytic solution (oblique edges only).
2. Buried rectangle vs brute-force area integration (exactly horizontal AND
   vertical edges) -- this is the case that regressed historically: the
   z1 == z2 limit of the edge formula is z*(th1 - th2), not zero, so skipping
   horizontal edges silently loses the slab term.
3. The surface-topped trapezoid used in the Shinshiro work vs brute force.

Only numpy is required.
"""

from __future__ import annotations

import numpy as np

from densigrav.section.talwani2d import G_SI, talwani_gz_polygon


def brute_gz(x_obs, z_obs, inside_fn, bbox, drho, h=5.0):
    """2-D area integration of 2*G*rho * z' / (x'^2 + z'^2)."""
    x0, x1, z0, z1 = bbox
    xs = np.arange(x0, x1, h) + h / 2
    zs = np.arange(z0, z1, h) + h / 2
    gx, gz = np.meshgrid(xs, zs)
    m = inside_fn(gx.ravel(), gz.ravel())
    px, pz = gx.ravel()[m], gz.ravel()[m]
    out = np.empty(len(np.atleast_1d(x_obs)))
    for i, (xo, zo) in enumerate(zip(np.atleast_1d(x_obs), np.atleast_1d(z_obs))):
        dx, dz = px - xo, pz - zo
        out[i] = 2.0 * G_SI * drho * np.sum(dz / (dx * dx + dz * dz)) * h * h * 1e5
    return out


def test_cylinder_matches_analytic():
    x0, z0, radius, drho = 0.0, 500.0, 120.0, 300.0
    th = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    verts = np.column_stack([x0 + radius * np.cos(th), z0 + radius * np.sin(th)])
    x = np.linspace(-1500, 1500, 31)
    z = np.zeros_like(x)
    got = talwani_gz_polygon(x, z, verts, drho)
    want = 2 * np.pi * G_SI * drho * radius**2 * z0 / ((x - x0) ** 2 + z0**2) * 1e5
    assert np.max(np.abs(got - want)) < 1e-4


def test_rectangle_matches_brute_force():
    xl, xr, zt, zb, drho = 4500.0, 6500.0, 100.0, 700.0, 78.0
    verts = np.array([[xl, zt], [xr, zt], [xr, zb], [xl, zb]])
    x = np.array([5000.0, 5500.0, 6047.0, 7000.0, 8273.0])
    z = np.zeros_like(x)
    got = talwani_gz_polygon(x, z, verts, drho)
    want = brute_gz(
        x,
        z,
        lambda px, pz: (px >= xl) & (px <= xr) & (pz >= zt) & (pz <= zb),
        (xl, xr, zt, zb),
        drho,
    )
    assert np.max(np.abs(got - want)) < 0.01  # mGal
    # interior value must be a sizable fraction of the infinite-slab limit
    slab = 2 * np.pi * G_SI * drho * (zb - zt) * 1e5
    assert got[1] > 0.5 * slab


def test_trapezoid_matches_brute_force():
    verts = np.array([[6047.0, 0.0], [10499.0, 0.0], [9464.0, 1457.0], [7082.0, 1457.0]])
    zt, zb = 0.0, 1457.0

    def inside(px, pz):
        f = np.clip((pz - zt) / (zb - zt), 0, 1)
        left = 6047.0 + f * (7082.0 - 6047.0)
        right = 10499.0 + f * (9464.0 - 10499.0)
        return (pz >= zt) & (pz <= zb) & (px >= left) & (px <= right)

    x = np.array([5000.0, 6047.0, 7000.0, 8273.0, 9250.0, 11000.0])
    z = np.zeros_like(x)
    got = talwani_gz_polygon(x, z, verts, drho := 78.0)
    want = brute_gz(x, z, inside, (6047.0, 10499.0, 0.0, 1457.0), drho)
    assert np.max(np.abs(got - want)) < 0.01  # mGal


def test_trapezoid_at_station_heights_matches_brute_force():
    """Stations at real (nonzero) heights above the body top -- the production
    configuration (z_obs = -elev_m, z positive down). Guards the observation-
    height convention: evaluating these stations at z=0 instead shifts the
    response by far more than the tolerance."""
    verts = np.array([[6047.0, 0.0], [10499.0, 0.0], [9464.0, 1457.0], [7082.0, 1457.0]])
    zt, zb = 0.0, 1457.0

    def inside(px, pz):
        f = np.clip((pz - zt) / (zb - zt), 0, 1)
        left = 6047.0 + f * (7082.0 - 6047.0)
        right = 10499.0 + f * (9464.0 - 10499.0)
        return (pz >= zt) & (pz <= zb) & (px >= left) & (px <= right)

    x = np.array([5000.0, 6047.0, 7000.0, 8273.0, 9250.0, 11000.0])
    z = -np.array([137.0, 250.0, 400.0, 598.0, 450.0, 155.0])  # station elevations
    got = talwani_gz_polygon(x, z, verts, drho := 78.0)
    want = brute_gz(x, z, inside, (6047.0, 10499.0, 0.0, 1457.0), drho)
    assert np.max(np.abs(got - want)) < 0.01  # mGal
    # the height must actually matter: z=0 evaluation differs measurably
    at_sea_level = talwani_gz_polygon(x, np.zeros_like(x), verts, drho)
    assert np.max(np.abs(got - at_sea_level)) > 0.1  # mGal
