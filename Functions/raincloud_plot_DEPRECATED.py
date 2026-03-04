''''
Script made by Constance Destais, 29/10/2025, with the use of ChatGPT-5, and heavily inspired by Antonis Nasioulas MATLAB function for "skyline" plots

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Rectangle
from matplotlib import rcParams
import matplotlib as mpl
import math
import warnings

# ------------------------------
# Helpers
# ------------------------------

def significance_stars(p):
    if np.isnan(p):
        return "n.a"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."

def _iqr(x):
    q75, q25 = np.nanpercentile(x, [75, 25])
    return q75 - q25

def _kde_on_support(x, n=200):
    """
    KDE similar spirit to MATLAB's ksdensity. Returns density and support.
    If all values are identical, returns a single flat density across min..max.
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.array([0.0, 0.0]), np.array([0.0, 0.0])

    xmin, xmax = np.min(x), np.max(x)
    if xmin == xmax:
        # Degenerate: all identical. Make a tiny support around the value.
        value = np.array([xmin, xmax])
        density = np.array([1.0, 1.0])
        return density, value

    # Scott's rule is close to your bandwidth formula; good default.
    kde = stats.gaussian_kde(x, bw_method="scott")
    value = np.linspace(xmin, xmax, n)
    density = kde(value)
    # Trim to within data range (parity with MATLAB line where they slice)
    keep = (value >= xmin) & (value <= xmax)
    value = value[keep]
    density = density[keep]
    # force endpoints to data extents
    value[0] = xmin
    value[-1] = xmax
    return density, value

def _paired_clean(a, b):
    """Return paired arrays with rows containing at least one NaN removed."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]

def _as_cell_like(DataCell):
    """
    Normalize DataCell to a list of 1-D float arrays (one per condition).
    Supports:
      A) [cond1, cond2, ...]              -> list/tuple of 1-D arrays (conditions)
      B) [data_2d, group_labels_1d]       -> matrix + categories (columns = subjects)
      C) 2D numpy array (rows = conditions)
      D) single 1-D array
    Returns: data_list, dataCat (or None)
    """
    import numpy as np

    dataCat = None

    # Detect the special (matrix + group_labels) ONLY when first is 2-D matrix
    # and second is 1-D labels with matching number of columns.
    if isinstance(DataCell, (list, tuple)) and len(DataCell) == 2:
        first, second = DataCell[0], DataCell[1]
        if isinstance(first, np.ndarray) and first.ndim == 2:
            glab = np.asarray(second).ravel()
            if glab.ndim == 1 and glab.size == first.shape[1]:
                raw = first
                dataCat = glab
            else:
                # not a valid (matrix, labels) pair -> treat as list of arrays
                raw = DataCell
        else:
            # not a matrix -> treat as list of arrays
            raw = DataCell
    else:
        raw = DataCell

    # Convert raw to list of 1-D arrays
    if isinstance(raw, np.ndarray):
        if raw.ndim == 1:
            data_list = [np.asarray(raw, dtype=float)]
        elif raw.ndim == 2:
            data_list = [np.asarray(row, dtype=float) for row in raw]
        else:
            raise ValueError("Data array must be 1-D or 2-D.")
    elif isinstance(raw, (list, tuple)):
        data_list = [np.asarray(r, dtype=float).ravel() for r in raw]
    else:
        raise ValueError("Unsupported DataCell type.")

    return data_list, dataCat


# ------------------------------
# Main plot function
# ------------------------------

def raincloud_plot(
    # necessary inputs
    plot_type,
    DataCell,
    my_colors,
    y_limits,
    *,
    # optional inputs with defaults
    ax=None, # existing axis to plot into; if None, creates new figure
    Title: str = "",
    LabelX: str = "",
    LabelY: str = "",
    x_tick_labels: list[str] = None, # labels for x ticks
    y_ticks: list = None, # numbers for y ticks
    reference_value: float = None, # value for horizontal midline
    figure_size: tuple =(8,5), # width, height, in inches
    font_size: float = 10, 
    font_name: str = "Arial",
    line_width: float = 1, # line width used for multiple elements
    sem_bar_width: float = 0.9, # width of SEM bars,as a proportion of the width of the mean's bar 
    highlight_connected_lines_in_dominant_direction: bool = True, # whether to color connecting lines based on direction
    dot_size: int = 30,            # size of individual data points
    dot_color: str = None,         # color for data points; single color or list length Nbar
    dot_alpha: float = 0.5,         # alpha for data points
    show_significance_stars: bool = False, 
    star_fontsize: float = None,       # fontsize when significant (*, **, ***)
    ns_fontsize: float = None,         # fontsize for 'n.s.' (and 'n.a')
    connect_linewidth: float = None,   # linewidth for connecting lines
    connect_alpha: float = 0.35,       # alpha for connecting lines (overrides defaults)
    # where the violins sit
    pair_gap: float = 0.30, # distance between the two halves of the violin in two-sided styles
    x_start: float = 1.0,          # where the first condition / first pair is centred
    dot_distance: float | None = None,  # distance from violin axis to dot column in x-units (bigger = further away) e.g. 0.07
    cloud_width: float = 0.75,   # >1 wider, <1 narrower
    x_margin: float | None = None   # margin to add on left/right of x-axis
):
    """
    Replicates the MATLAB skylineplot_Antonis in Python/Matplotlib.

    Parameters mirror the MATLAB function, with small Pythonic tweaks.

    plot_type: int
        1: one-sided violins
        2: one-sided violins, connect participants
        3: two-sided violins
        4: two-sided violins, connect participants
        5: one factor / only boxes (mean, sem, CI)
        6: two factors / only boxes (mean, sem, CI)
    DataCell:
        - 2D array where rows are conditions, or
        - list of 1-D arrays (one per condition), or
        - [data, group_labels] to split mean bars by category within each condition
    my_colors:
        list/array of RGB tuples in [0,1], one per condition
    """

    data_list, dataCat = _as_cell_like(DataCell)
    Nbar = len(data_list)
    if my_colors is None or len(my_colors) == 0:
        my_colors = [(0, 0, 0)] * Nbar
    if len(my_colors) < Nbar:
        # Repeat last color if fewer supplied
        my_colors = list(my_colors) + [my_colors[-1]] * (Nbar - len(my_colors))

    

    # ----- defaults for options -----
    # point colors: allow a single color or per-condition list
    if dot_color is None:
        dot_colors = my_colors
    else:
        # if a single (r,g,b) or (r,g,b,a) -> use for all; if list -> per condition
        if isinstance(dot_color, (list, tuple)) and len(dot_color) == Nbar and \
        isinstance(dot_color[0], (list, tuple, np.ndarray)):
            dot_colors = dot_color
        else:
            dot_colors = [dot_color] * Nbar

    # connecting lines linewidth default mirrors prior behavior
    if connect_linewidth is None:
        connect_linewidth = max(line_width / 3.0, 0.5)
    else:
        connect_linewidth = connect_linewidth

    # star sizes: by default, n.s. uses font_size, sig uses font_size+1 (as in your code)
    if star_fontsize is None:
        star_fontsize = font_size + 1
    if ns_fontsize is None:
        ns_fontsize = font_size

    # base distance between datapoints and violin
    # distance from center to each cloud (so gap=1)
    pair_distance_half = pair_gap / 2.0

    # after you know cloud_width (and maybe effective_width)
    internal_margin = 0.5 * cloud_width  # or 0.6, 0.7… whatever looks good

    xmin = min(xRefs_log) - internal_margin
    xmax = max(xRefs_log) + internal_margin
    ax.set_xlim([xmin, xmax])

    # -------------------------------------------

    with mpl.rc_context(rc={'font.family': font_name, 'font.size': font_size}):

        # set font settings that should be compatible with Inkscape ("This avoids some of the weird Type 3 font behavior that Inkscape doesn’t always enjoy.")
        mpl.rcParams["pdf.fonttype"] = 42  # TrueType fonts in PDF
        mpl.rcParams["ps.fonttype"] = 42   # same for EPS (if ever)
        mpl.rcParams["text.usetex"] = False

        if ax is None:
            fig, ax = plt.subplots(figsize=figure_size)

        #mpl.rcParams['font.family'] = font_name
        ax.set_title(Title, fontsize=font_size, fontname=font_name)
        ax.set_xlabel(LabelX, fontsize=font_size, fontname=font_name)
        ax.set_ylabel(LabelY, fontsize=font_size, fontname=font_name)

        # constants
        confidence_interval = 0.95
        alpha_lines = connect_alpha
        color_dominant = (1.0, 0.5529, 0.3608, alpha_lines)   # orange (dominant)
        color_non_dominant = (0.4, 0.4, 0.4, alpha_lines)      # gray (non-dominant)
        color_all = (0.6, 0.6, 0.6, alpha_lines)         # same color for all when not highlighting

        # x margin default
        if x_margin is None:
            x_margin = 0.5 * cloud_width

        # dot–violin distance default 
        # If user doesn't specify, use 1/10 of the bar width (your old behavior)
        if dot_distance is None:
            dot_distance = cloud_width / 10.0

        # Midline 
        if reference_value is not None:
            ax.axhline(reference_value, linestyle='--', linewidth=1.5,
                        color="grey", alpha=0.5, zorder=0)


        if plot_type in (2, 4):
            # require matched sizes for paired lines inside each pair
            if plot_type == 2:
                if len(data_list) < 2 or len(data_list[0]) != len(data_list[1]):
                    raise AssertionError("Type 2 requires two conditions with the same number of points.")
            elif plot_type == 4:
                # Requires pairs (1,2) and (3,4) with equal lengths
                if len(data_list) < 4 or (len(data_list[0]) != len(data_list[1])) or (len(data_list[2]) != len(data_list[3])):
                    raise AssertionError("Type 4 requires (1,2) and (3,4) to have the same number of points within each pair.")

        # to store x positions used (for significance bars later)
        xRefs_log = []

        # to store x positions of dots (for paired lines later)
        dot_anchor_x = [None] * Nbar

        # convenience: means per condition (for dominant-direction logic)
        # Compute robustly even if arrays have different lengths
        means = np.array([np.nanmean(d) if len(d) else np.nan for d in data_list])

        # iterate conditions
        for n in range(Nbar):

            # prep data for violins
            DataMatrix = np.asarray(data_list[n]).ravel()
            keep = ~np.isnan(DataMatrix)
            DataMatrix = DataMatrix[keep]
            Nsub = DataMatrix.size

            curve = np.nanmean(DataMatrix) if Nsub > 0 else np.nan
            sem = (np.nanstd(DataMatrix, ddof=1) / np.sqrt(Nsub)) if Nsub > 1 else 0.0
            conf = stats.t.ppf(1 - 0.5 * (1 - confidence_interval), max(Nsub - 1, 1))

            # KDE for violin
            density, value = _kde_on_support(DataMatrix)

            # violin width scaling
            ''' 
            if plot_type == 4:
                cloud_width_compression_per_plot_type = 2.5
            else:
                cloud_width_compression_per_plot_type = 1.6

            effective_width = cloud_width / cloud_width_compression_per_plot_type
            '''

            # Avoid div/0: if density is constant ones, max=1; otherwise use max
            max_density = np.max(density) if density.size else 1.0
            density_to_x_scale = cloud_width  / (max_density if max_density > 0 else 1.0)

            # positioning
            if plot_type == 1:
                cloud_orientation = 1
                dot_side_factor, dot_jitter_factor = 1, 1
                # conditions at x_start, x_start+1, ...
                xRef = x_start + n
                xPos = xRef
            elif plot_type == 2:
                
                # type 2: paired one-sided half-violins so you can connect participants
                dot_side_factor, dot_jitter_factor = 1, -10
                if Nbar == 2:
                    # Special paired case: two conditions symmetric around x_start
                    pair_center = x_start              # midpoint between the two
                    side = -1 if n == 0 else 1        # first condition on the left, second on the right
                    # Make the half-violins face each other:
                    #  left condition (side=-1) => cloud_orientation=+1 (extends rightwards)
                    #  right condition (side=+1) => cloud_orientation=-1 (extends leftwards)
                    cloud_orientation = side
                    xRef = pair_center + side * pair_distance_half
                    xPos = xRef
                else:
                    # General (non-paired) case: simple one-sided violins spaced from x_start
                    cloud_orientation = 1
                    xRef  = x_start + n
                    xPos  = xRef

            elif plot_type in (3, 6):
                cloud_orientation = 1 - 2 * ((n + 1) % 2)   # left/right half
                dot_side_factor, dot_jitter_factor = -1, -1

                pair_index  = n // 2            # 0,0,1,1,2,2,...
                pair_center = x_start + pair_index
                side        = -1 if (n % 2 == 0) else +1   # left then right

                xRef = pair_center + side * pair_distance_half
                # For the left half, rectangle starts at xRef; for right, shift left by its width?
                xPos = xRef - (cloud_width  if side < 0 else 0.0)
            elif plot_type == 4:
                cloud_orientation = 1 - 2 * ((n + 1) % 2)
                dot_side_factor, dot_jitter_factor = 1/3, -1

                pair_index  = n // 2              # 0,0,1,1,2,2,...
                pair_center = x_start + pair_index
                side        = -1 if (n % 2 == 0) else +1   # left/right

                xRef = pair_center + side * pair_distance_half
                xPos = xRef - (cloud_width  if side < 0 else 0.0)
            elif plot_type == 5:
                cloud_orientation = 1
                # centring each box around x_start+n, then shifting left half-width
                xRef = (x_start + n) - cloud_width / (2 * cloud_width_compression_per_plot_type)
                xPos = xRef
            else:
                raise ValueError("plot_type must be 1..6")

            # Violin & dots & CI stripe (skip for 5 and 6)
            if plot_type not in (5, 6) and Nsub > 0:
                # semi-violin fill
                xv = np.concatenate([[xRef], xRef + cloud_orientation * density * density_to_x_scale, [xRef]])
                yv = np.concatenate([[value[0]], value, [value[-1]]])
                ax.fill(xv, yv, color=my_colors[n], alpha=0.2, lw=0)

                # jitter strength from density to avoid overplot
                if value.size > 1:
                    jitterstrength = np.interp(DataMatrix, value, density * density_to_x_scale)
                else:
                    jitterstrength = np.full_like(DataMatrix, fill_value=density[0] * density_to_x_scale)

                # jitter layout
                if plot_type in (1, 3):
                    # narrow, uniform strip around a central x — NOT density-shaped
                    strip_width = 0.05  # tweak if you want wider/narrower
                    # reproducible jitter without global RNG changes:
                    rng_local = np.random.default_rng(0)
                    jitter = rng_local.uniform(-strip_width, strip_width, size=Nsub) / (cloud_width/2 - cloud_width/10)
                else:
                    jitter = np.zeros(Nsub)

                # horizontal placement of dots 
                # dot_distance = base offset from violin axis (same for all types)
                # dot_side_factor, cloud_orientation decide which side (left/right) and direction
                anchor_x = xRef - dot_side_factor * cloud_orientation * dot_distance
                # jitter spreads them slightly along x for types 1 & 3; zero for 2 & 4
                x_dots = anchor_x - dot_jitter_factor * cloud_orientation * jitter * (cloud_width / 2.0 - cloud_width / 10.0)
                ax.scatter(
                    x_dots,
                    DataMatrix,
                    s=dot_size,
                    color=dot_colors[n],
                    marker="o",
                    alpha=dot_alpha,
                    linewidths=0,
                    zorder=3,
                )


                # cache the anchor for this condition (will be used for the lines connecting dots)
                dot_anchor_x[n] = anchor_x

                # bold CI stripe inside the violin
                if value.size > 1 and not np.isnan(curve):
                    ystripe = np.arange(curve - sem * conf, curve + sem * conf, 0.0001)
                    if ystripe.size > 0:
                        dstripe = np.interp(ystripe, value, density * density_to_x_scale,
                                            left=0.0, right=0.0)
                        xv2 = np.concatenate([[xRef], xRef + cloud_orientation * dstripe, [xRef]])
                        yv2 = np.concatenate([[ystripe[0]], ystripe, [ystripe[-1]]])
                        ax.fill(xv2, yv2, color=my_colors[n], alpha=0.4, lw=0)


            # Mean bar + error bar + rectangle (with or without categories)
            if dataCat is None:
                if not np.isnan(curve):
                    # horizontal mean bar
                    xMean = [xRef, xRef + cloud_orientation * cloud_width ]
                    yMean = [curve, curve]
                    ax.plot(xMean, yMean, "-", lw=line_width, color=(0, 0, 0, 1))

                    # errorbar (SEM)
                    ax.errorbar(
                        xRef + cloud_orientation * cloud_width / (2 * cloud_width_compression_per_plot_type),
                        curve,
                        yerr=sem,
                        color=(0, 0, 0, 1),
                        fmt="none",
                        elinewidth=line_width,
                        capsize = sem_bar_width,
                        zorder=4,
                    )

                    # CI rectangle
                    rect = Rectangle(
                        (xPos, curve - sem * conf),  # lower-left
                        cloud_width ,                  # width
                        2 * sem * conf,              # height
                        fill=False,
                        edgecolor=my_colors[n],
                        linewidth=line_width,
                        zorder=2,
                        clip_on=False
                    )
                    ax.add_patch(rect)

            else:
                # split rectangle horizontally by categories within condition n
                # Only use columns where group label is available and data is valid
                full = data_list[n]
                # dataCat length is max length; trim/pad data accordingly
                L = len(dataCat)
                if len(full) < L:
                    # pad with NaNs to length L (rare)
                    full = np.pad(full, (0, L - len(full)), constant_values=np.nan)
                vals = np.asarray(full)[:L]
                glab = dataCat.copy()
                mask = ~np.isnan(vals) & ~np.isnan(glab)
                vals = vals[mask]
                glab = glab[mask]
                if vals.size == 0:
                    pass
                else:
                    cats = np.unique(glab)
                    w = (cloud_width ) / max(len(cats), 1)
                    # For two-sided styles, adjust xRefCat like MATLAB
                    xRefCat = xRef if plot_type in (1, 2, 5) else (xRef - (( (n + 1) % 2) ) * cloud_width )

                    for i, c in enumerate(cats):
                        data_c = vals[glab == c]
                        if data_c.size == 0:
                            continue
                        curveDC = np.nanmean(data_c)
                        semDC = np.nanstd(data_c, ddof=1) / np.sqrt(np.sum(~np.isnan(data_c)))

                        # mean sub-bar
                        x0 = xRefCat + w * i
                        x1 = xRefCat + w * (i + 1)
                        ax.plot([x0, x1], [curveDC, curveDC], "-", lw=line_width, color=(0, 0, 0, 1))

                        # errorbar
                        ax.errorbar(
                            (x0 + x1) / 2.0,
                            curveDC,
                            yerr=semDC,
                            color=(0, 0, 0, 1),
                            fmt="none",
                            elinewidth=line_width,
                            capsize=sem_bar_width,
                            zorder=4,
                        )
                        # Light label above (category code)
                        ax.text(
                            (x0 + x1) / 2.0,
                            curveDC + semDC + 0.01 * (y_limits[1] - y_limits[0]),
                            str(c),
                            fontsize=font_size,
                            fontname=font_name,
                            ha="center",
                            va="baseline",
                            color=(0, 0, 0),
                        )

            xRefs_log.append(xRef)

        # --- Unified axis/ticks/tick labels for all types (do this ONCE, after the loop) ---
        ax.set_ylim([y_limits[0], y_limits[1]])

        if y_ticks is not None:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(t) for t in y_ticks],
                            fontfamily=font_name,
                            fontsize=font_size)

        if plot_type in (3, 4):
            # Use actual x-positions you plotted at
            xmin = min(xRefs_log) - x_margin
            xmax = max(xRefs_log) + x_margin
            ax.set_xlim([xmin, xmax])

            # Decide tick mode:
            # - If user passed Nbar labels -> per-condition ticks (one tick per half-violin)
            # - Else (None or Nbar/2 labels) -> per-pair ticks (centered)
            if x_tick_labels is not None and len(x_tick_labels) == Nbar:
                # 4 (or Nbar) ticks: under each half-violin
                ax.set_xticks(xRefs_log)
                ax.set_xticklabels(x_tick_labels)
            else:
                # 2 (or Nbar/2) ticks: centered under each pair
                pair_centers = [
                    0.5 * (xRefs_log[i] + xRefs_log[i + 1])
                    for i in range(0, Nbar, 2)
                    if i + 1 < Nbar
                ]
                if x_tick_labels is None or len(x_tick_labels) != len(pair_centers):
                    x_tick_labels = [str(i + 1) for i in range(len(pair_centers))]
                ax.set_xticks(pair_centers)
                ax.set_xticklabels(x_tick_labels)
        else:
            # Types 1/2/5/6:
            # always put ticks exactly where the clouds actually are (xRefs_log)
            xmin = min(xRefs_log) - x_margin
            xmax = max(xRefs_log) + x_margin
            ax.set_xlim([xmin, xmax])

            ax.set_xticks(xRefs_log)

            if x_tick_labels is None or len(x_tick_labels) != Nbar:
                # no or mismatched labels → leave ticks unlabeled
                ax.set_xticklabels([])
            else:
                # label each cloud in the order of DataCell / data_list
                ax.set_xticklabels(x_tick_labels)

        # ---- Connecting lines for type 2 and type 4 ----
        if plot_type in (2, 4):
            if plot_type == 2 and Nbar == 2:
                # One pair: (0,1)
                pairs = [(0, 1)]
            elif plot_type == 4:
                # Pairs: (0,1), (2,3), ...
                pairs = [(i, i+1) for i in range(0, Nbar, 2) if i+1 < Nbar]
            else:
                pairs = []

            if pairs:
                # Draw pairwise lines
                for i, j in pairs:
                    x_left, x_right = dot_anchor_x[i], dot_anchor_x[j]
                    a, b = _paired_clean(data_list[i], data_list[j])
                    for ai, bi in zip(a, b):
                        if highlight_connected_lines_in_dominant_direction and not (np.isnan(means[i]) or np.isnan(means[j])):
                            col = color_dominant if np.sign(ai - bi) == np.sign(means[i] - means[j]) else color_non_dominant
                        else:
                            col = color_all
                        ax.plot([x_left, x_right], [ai, bi], color=col, lw=connect_linewidth, zorder=2)

            else:
                # Optional: type 2 with more than two conditions -> polylines
                xs = [dot_anchor_x[k] for k in range(Nbar)]
                min_len = min(len(d) for d in data_list)
                for s in range(min_len):
                    yy = [data_list[k][s] for k in range(Nbar)]
                    if any(np.isnan(yy)):
                        continue
                    ax.plot(xs, yy, color=(0, 0, 0, alpha_lines), lw=connect_linewidth, zorder=2)

        # --- Significance stars (types 2 and 4), drawn above the axes without changing ylim ---
        from matplotlib.transforms import blended_transform_factory

        if show_significance_stars and plot_type in (2, 4) and len(xRefs_log) >= 2:
            # Which pairs to test
            pairs = [(0, 1)] if plot_type == 2 else ([(0, 1), (2, 3)] if len(xRefs_log) >= 4 else [])

            # y in axes coordinates (0..1), x in data coordinates
            trans = blended_transform_factory(ax.transData, ax.transAxes)

            # How far above the top to draw the line(relative to axes height)
            star_offset_rel = 0.02  # 2% above top; bump if you need more space

            for (a_idx, b_idx) in pairs:
                # x positions for the bar (data coords)
                x_bar = [xRefs_log[a_idx], xRefs_log[b_idx]]
                x_stars = 0.5 * (x_bar[0] + x_bar[1])

                # paired t-test -> stars string
                a, b = _paired_clean(data_list[a_idx], data_list[b_idx])
                if a.size > 1 and b.size > 1:
                    _, p = stats.ttest_rel(a, b, nan_policy="omit")
                else:
                    p = np.nan
                stars = significance_stars(p)

                # draw the horizontal bar just above the top spine
                y_bar_axes = 1.0 + star_offset_rel
                ax.plot(
                    x_bar,
                    [y_bar_axes, y_bar_axes],
                    color=(0, 0, 0, 1),
                    lw=line_width,
                    transform=trans,
                    clip_on=False,
                    zorder=5,
                )

                # stars a touch higher; keep your font-size logic
                fsize = star_fontsize if stars in ("*", "**", "***") else ns_fontsize
                y_text_axes = y_bar_axes + (0.01 if stars == "n.a" else -0.050)
                ax.text(
                    x_stars,
                    y_text_axes,
                    stars,
                    fontsize=fsize,
                    fontname=font_name,
                    ha="center",
                    va="bottom",
                    transform=trans,
                    clip_on=False,
                    zorder=6,
                )


        # --- grid and ticks ---
        ax.grid(False)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color('black')

        # make ticks point inward, and set their length/width
        ax.tick_params(axis='both', which='major', length=4, width=1,
                    direction='in', labelsize=font_size)
        
        # Force tick labels (axis numbers) to use the same font as axis labels
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontname(font_name)
            lbl.set_fontsize(font_size)   
        
        #plt.close(fig)

        if ax is not None: 
            return ax.figure, ax 
        else: 
            return fig, ax




#%%# ---------------------------------------------------------------------------------------------------------------------
# Four demos : fake datasets & example calls to the skyline function
# ------------------------------------------------------------------------------------------------------------------------
'''

# from Functions.raincloud_plotting import raincloud_plot 
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    
    # --- Example 1: four one-sided ---
    # tiny demo data 
    rng = np.random.default_rng(7)
    d1 = rng.normal(0.55, 0.18, 25) * 0.7
    d2 = rng.normal(0.50, 0.20, 22) * 0.8
    d3 = rng.normal(0.45, 0.15, 28) * 0.9
    d4 = rng.normal(0.60, 0.22, 24) * 0.6

    # colors (R,G,B in [0,1])
    red   = (0.84, 0.16, 0.16)
    green = (0.16, 0.60, 0.32)
    blue  = (0.10, 0.40, 0.80)
    gold  = (0.86, 0.66, 0.20)

    # fig, ax = plt.subplots(figsize=(9, 5))
    fig4, ax4 = raincloud_plot(
        plot_type=1,                         # one-sided violins
        DataCell=[d1, d2, d3, d4],           # four conditions, no category split
        my_colors=[red, green, blue, gold],
        Yinf=0.0, Ysup=1.0,
        xRefInput=np.nan,
        font_size=14,
        Title="Four one-sided violins",
        LabelX="Condition", LabelY="Value",
        x_tick_labels=["C1", "C2", "C3", "C4"],
        dot_size=80,
        sem_bar_width=12,
        line_width=2,
        highlight_connected_lines_in_dominant_direction=True, 
        font="Arial",
        ax=ax
    )
    plt.tight_layout()
    plt.show()



    # --- Example 2: type 2 (two one-sided violins with paired lines) ---
    # Small test data like in the MATLAB comments ---
    d1 = rng.normal(0.5, 0.2, 20) * 0.5
    d2 = rng.normal(0.5, 0.2, 20) * 0.7
    d3 = rng.normal(0.5, 0.2, 20) * 0.8
    d4 = rng.normal(0.5, 0.2, 20)

    ymin, ymax = 0.0, 1.0
    fontsize = 14
    x_ref_input = np.nan
    red = (0.84, 0.16, 0.16)
    green = (0.16, 0.6, 0.32)

    # fig1, ax1 = plt.subplots(figsize=(7, 5))
    fig1, ax1 = raincloud_plot(
        plot_type=2,
        DataCell=[d1, d2],
        my_colors=[red, green],
        Yinf=ymin,
        Ysup=ymax,
        xRefInput=x_ref_input,
        font_size=fontsize,
        Title="Type 2 — paired one-sided violins",
        LabelX="Condition",
        LabelY="Value",
        x_tick_labels=["A", "B"],
        dot_size=80,
        sem_bar_width=12,
        line_width=2,
        highlight_connected_lines_in_dominant_direction=True,
    )
    plt.tight_layout()

    

    # --- Example 3: type 4 (two two-sided paired violins with lines) ---
    # Small test data like in the MATLAB comments ---
    d1 = rng.normal(0.5, 0.2, 20) * 0.5
    d2 = rng.normal(0.5, 0.2, 20) * 0.7
    d3 = rng.normal(0.5, 0.2, 20) * 0.8
    d4 = rng.normal(0.5, 0.2, 20)

    ymin, ymax = 0.0, 1.0
    fontsize = 14
    x_ref_input = np.nan
    red = (0.84, 0.16, 0.16)
    green = (0.16, 0.6, 0.32)

    fig2, ax2 = raincloud_plot(
        plot_type=4,
        DataCell=[d1, d2, d3, d4],
        my_colors=[red, green, red, green],
        Yinf=ymin,
        Ysup=ymax,
        xRefInput=x_ref_input,
        font_size=fontsize,
        Title="Type 4 — paired two-sided violins",
        LabelX="Pairs",
        LabelY="Value",
        x_tick_labels=["L1", "G1", "L2", "G2"],
        dot_size=80,
        sem_bar_width=14,
        line_width=2,
        highlight_connected_lines_in_dominant_direction=True,
    )
    plt.tight_layout()



    # --- Example 4: one sided, with categories within rectangles (like dataCat) ---
    # Small test data like in the MATLAB comments ---
    d1 = rng.normal(0.5, 0.2, 20) * 0.5
    d2 = rng.normal(0.5, 0.2, 20) * 0.7
    d3 = rng.normal(0.5, 0.2, 20) * 0.8
    d4 = rng.normal(0.5, 0.2, 20)

    ymin, ymax = 0.0, 1.0
    fontsize = 14
    x_ref_input = np.nan
    red = (0.84, 0.16, 0.16)
    green = (0.16, 0.6, 0.32)

    # Make group labels per column (subjects). Here, 0/1 groups.
    group_labels = np.array([0]*10 + [1]*10)
    Dcat = np.vstack([d1, d2])  # two conditions, same length
    #fig3, ax3 = plt.subplots(figsize=(7, 5))
    fig3, ax3 = raincloud_plot(
        plot_type=1,
        DataCell=[Dcat, group_labels],  # pass as [data, group_labels]
        my_colors=[red, green],
        Yinf=ymin,
        Ysup=ymax,
        Title="Type 1 — one-sided with category split",
        LabelX="Condition",
        LabelY="Value",
        x_tick_labels=["C1", "C2"],
        dot_size=70,
        sem_bar_width=10,
    )
    plt.tight_layout()

    plt.show()


'''