import numpy as np
from matplotlib.patches import Rectangle
from scipy import stats

# ------------------------------
# Helpers
# ------------------------------


def _kde_on_support(x, n: int = 200):
    """
    KDE similar to MATLAB's ksdensity. Returns (density, support).

    - If all values are identical, returns flat density at that value.
    - If x is empty or all NaN, returns trivial zeros.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([0.0, 0.0]), np.array([0.0, 0.0])

    xmin, xmax = np.min(x), np.max(x)
    if xmin == xmax:
        value = np.array([xmin, xmax])
        density = np.array([1.0, 1.0])
        return density, value

    kde = stats.gaussian_kde(x, bw_method="scott")
    value = np.linspace(xmin, xmax, n)
    density = kde(value)

    # Restrict to [xmin, xmax] just in case
    keep = (value >= xmin) & (value <= xmax)
    value = value[keep]
    density = density[keep]

    # Force exact endpoints
    value[0] = xmin
    value[-1] = xmax
    return density, value


def _paired_clean(a, b):
    """Return paired arrays with any pair containing NaN removed."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def _draw_one_semi_violin(
    ax,
    data,
    *,
    xRef: float,
    color,
    dot_color,
    cloud_width: float,
    cloud_orientation: float,
    dot_distance: float,
    dot_size: int,
    dot_alpha: float,
    line_width: float,
    sem_cap_width: float,
    confidence_interval: float = 0.95,
):
    """
    Draw a single half-violin + dots + mean/SEM/CI for one condition.

    Returns a dict with basic stats and the x position used to anchor dots.
    """

    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    Nsub = data.size

    if Nsub == 0:
        # nothing to draw visually, but keep info for callers
        return {
            "Nsub": 0,
            "curve": np.nan,
            "sem": 0.0,
            "anchor_x": xRef,
        }

    curve = np.nanmean(data)
    sem = np.nanstd(data, ddof=1) / np.sqrt(Nsub) if Nsub > 1 else 0.0
    df = max(Nsub - 1, 1)
    conf = stats.t.ppf(1 - 0.5 * (1 - confidence_interval), df)

    density, value = _kde_on_support(data)
    max_density = np.max(density) if density.size else 1.0
    density_to_x_scale = cloud_width / (max_density if max_density > 0 else 1.0)

    # semi-violin
    xv = np.concatenate(
        [
            [xRef],
            xRef + cloud_orientation * density * density_to_x_scale,
            [xRef],
        ]
    )
    yv = np.concatenate([[value[0]], value, [value[-1]]])
    ax.fill(xv, yv, color=color, alpha=0.2, lw=0, zorder=1)

    # CI stripe
    if value.size > 1 and not np.isnan(curve):
        y_low = curve - sem * conf
        y_high = curve + sem * conf
        ystripe = np.linspace(y_low, y_high, 200)
        dstripe = np.interp(
            ystripe,
            value,
            density * density_to_x_scale,
            left=0.0,
            right=0.0,
        )
        xv2 = np.concatenate(
            [[xRef], xRef + cloud_orientation * dstripe, [xRef]]
        )
        yv2 = np.concatenate([[ystripe[0]], ystripe, [ystripe[-1]]])
        ax.fill(xv2, yv2, color=color, alpha=0.4, lw=0, zorder=2)

    # dots: on the side opposite the bulge
    anchor_x = xRef - cloud_orientation * dot_distance
    x_dots = np.full_like(data, anchor_x)
    ax.scatter(
        x_dots,
        data,
        s=dot_size,
        color=dot_color,
        marker="o",
        alpha=dot_alpha,
        linewidths=0,
        zorder=3,
    )

    # mean bar along bulge side
    xMean0 = xRef
    xMean1 = xRef + cloud_orientation * cloud_width
    ax.plot(
        [xMean0, xMean1],
        [curve, curve],
        "-",
        lw=line_width,
        color=(0, 0, 0, 1),
        zorder=4,
    )

    # SEM errorbar at midpoint of that bar
    x_err = (xMean0 + xMean1) / 2.0
    ax.errorbar(
        x_err,
        curve,
        yerr=sem,
        color=(0, 0, 0, 1),
        fmt="none",
        elinewidth=line_width,
        capthick=line_width,
        capsize=sem_cap_width,
        zorder=5,
    )

    # CI rectangle (outline only)
    rect_x = min(xMean0, xMean1)
    rect_width = abs(xMean1 - xMean0)
    rect = Rectangle(
        (rect_x, curve - sem * conf),
        rect_width,
        2 * sem * conf,
        fill=False,
        edgecolor=color,
        linewidth=line_width,
        zorder=2,
        clip_on=False,
    )
    ax.add_patch(rect)

    return {
        "Nsub": Nsub,
        "curve": curve,
        "sem": sem,
        "anchor_x": anchor_x,
        "rect_width": rect_width,
    }


