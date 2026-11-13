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


# --- bodies above / straddling the observation height -------------------------
# Regression guard: the forward used to fold each vertex angle into [0, pi)
# separately, which is only valid when the whole polygon lies below the
# observation point. A body reaching above a station (e.g. a trapezoid whose top
# follows the ground surface, or a sheet above a valley station) then gave wrong
# values (e.g. -7.3 mGal instead of +0.06 for the trapezoid below).

STRADDLE = np.array([[2000.0, -300.0], [4000.0, -300.0], [3500.0, 800.0], [2500.0, 800.0]])


def _inside_straddle(px, pz):
    f = np.clip((pz + 300.0) / 1100.0, 0, 1)
    return (pz >= -300.0) & (pz <= 800.0) & (px >= 2000.0 + 500.0 * f) & (px <= 4000.0 - 500.0 * f)


def test_body_straddling_station_height_matches_brute_force():
    """Stations beside a body whose top is above them (oblique edges cross z_obs)."""
    x = np.array([0.0, 1900.0, 5000.0, 4300.0])
    z = np.array([-137.0, -137.0, -500.0, 200.0])
    got = talwani_gz_polygon(x, z, STRADDLE, drho := 78.0)
    want = brute_gz(x, z, _inside_straddle, (2000.0, 4000.0, -300.0, 800.0), drho, h=2.0)
    assert np.max(np.abs(got - want)) < 0.005  # mGal
    assert got[0] > 0  # most of the mass is below the station


def test_station_inside_body_matches_brute_force():
    """Observation point inside the body (integrable 1/r singularity -> looser tolerance)."""
    x = np.array([3001.3, 3777.7])
    z = np.array([-137.3, 100.7])
    got = talwani_gz_polygon(x, z, STRADDLE, drho := 78.0)
    want = brute_gz(x, z, _inside_straddle, (2000.0, 4000.0, -300.0, 800.0), drho, h=2.0)
    assert np.max(np.abs(got - want)) < 0.02  # mGal


def test_slab_above_station_is_negative_slab_value():
    """A wide sheet entirely above the station pulls upward: gz -> -2*pi*G*rho*t."""
    t, drho = 300.0, 78.0
    verts = np.array([[-1e6, -400.0], [1e6, -400.0], [1e6, -400.0 + t], [-1e6, -400.0 + t]])
    got = talwani_gz_polygon(np.zeros(1), np.zeros(1), verts, drho)[0]
    want = -2 * np.pi * G_SI * drho * t * 1e5
    assert abs(got - want) < 1e-3 * abs(want)


def test_split_body_is_additive_across_station_height():
    """Cutting a body along the station height must not change the total."""
    drho = 78.0
    whole = np.array([[2000.0, -300.0], [4000.0, -300.0], [4000.0, 800.0], [2000.0, 800.0]])
    upper = np.array([[2000.0, -300.0], [4000.0, -300.0], [4000.0, -137.0], [2000.0, -137.0]])
    lower = np.array([[2000.0, -137.0], [4000.0, -137.0], [4000.0, 800.0], [2000.0, 800.0]])
    x = np.array([0.0, 1500.0, 3000.0, 4600.0])
    z = np.full_like(x, -137.0)  # stations exactly at the cut (on the line of a horizontal edge)
    total = talwani_gz_polygon(x, z, whole, drho)
    parts = talwani_gz_polygon(x, z, upper, drho) + talwani_gz_polygon(x, z, lower, drho)
    assert np.max(np.abs(total - parts)) < 1e-9
    # the upper part alone pulls up, the lower part pulls down
    assert np.all(talwani_gz_polygon(x, z, upper, drho) < 0)
    assert np.all(talwani_gz_polygon(x, z, lower, drho) > 0)


def test_vertex_order_and_boundary_points_are_finite():
    drho = 78.0
    x = np.array([2000.0, 3000.0, 2250.0, 0.0])  # at a vertex, on the top edge, on a wall, outside
    z = np.array([-300.0, -300.0, 250.0, -137.0])
    got = talwani_gz_polygon(x, z, STRADDLE, drho)
    assert np.all(np.isfinite(got))
    assert np.max(np.abs(got - talwani_gz_polygon(x, z, STRADDLE[::-1], drho))) < 1e-9
