import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams
import matplotlib as mpl
from scipy import stats
from Functions.set_axes_size import set_axes_size_inches
from Functions.raincloud_helpers import _kde_on_support, _paired_clean, _draw_one_semi_violin
from Functions.significance_stars import significance_stars

# ------------------------------
# Main function: type-2 only
# ------------------------------

def raincloud_paired_two_conditions(
    data,
    *,
    my_colors: tuple[list[float]] = ((0.2, 0.4, 0.8), (0.8, 0.4, 0.2)),
    y_limits: list[float] = None,
    ax=None,
    reference_value: float = None,
    # axis labels & ticks
    title: str = "",
    label_x: str = "",
    label_y: str = "",
    x_tick_labels=None,   # list of 2 labels or None
    y_ticks=None,
    # figure + fonts
    figure_size=(6, 4), # width, height of axis, in inches
    font_size: float = 10,
    font_name: str = "Arial",
    # geometry / layout
    cloud_width: float = 0.75,
    pair_gap: float = 0.30,      # distance between the two violins
    x_margin: float | None = None,
    # dots
    dot_distance: float | None = None,  # lateral offset of dots from violin axis
    dot_size: int = 30,
    dot_color=None,
    dot_alpha: float = 0.5,
    # mean / error bars
    line_width: float = 1.0,
    sem_cap_width: float = 5.0,
    # connecting lines
    connect_linewidth: float | None = None,
    connect_alpha: float = 0.35,
    highlight_connected_lines_in_dominant_direction: bool = True,
    # significance stars
    show_significance_stars: bool = False,
    star_fontsize: float | None = None,
    ns_fontsize: float | None = None,
):
    """
    Raincloud-style plot for exactly TWO paired conditions.

    Each condition is shown as a one-sided (half) violin, facing the other.
    Individual points are shown on the opposite side of each violin.
    Paired observations are connected with lines, optionally colored by
    whether they follow the dominant mean difference direction.
    """

    # ---- validate and prepare data ----
    a = np.asarray(data[0], dtype=float).ravel()
    b = np.asarray(data[1], dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("data_a and data_b must have the same length (paired design).")
    if len(data) != 2:
        raise ValueError("data must contain exactly two conditions for paired design.")

     # ---- prepare plotting parameters ----
    data_list = [a, b]
    N_conditions = 2

    # Remove NaNs only for global stats if needed, but keep full arrays for plotting
    all_valid = np.concatenate([d[~np.isnan(d)] for d in data_list])
    if all_valid.size == 0:
        raise ValueError("Both conditions are empty or all NaN.")

    # Colors
    if len(my_colors) > 2:
        # Use only first two colors
        my_colors = list(my_colors)[:2]

    # Dot colors: single or per-condition
    if dot_color is None:
        dot_colors = my_colors
    else:
        # If given as a single color-like thing, use for both
        if isinstance(dot_color, (tuple, list)) and len(dot_color) == 2 \
           and isinstance(dot_color[0], (tuple, list, np.ndarray)):
           dot_colors = list(dot_color)
        else:
            dot_colors = [dot_color, dot_color]

    # Linewidth for connecting lines
    if connect_linewidth is None:
        connect_linewidth = max(line_width / 3.0, 0.5)

    # Star font sizes
    if star_fontsize is None:
        star_fontsize = font_size + 1
    if ns_fontsize is None:
        ns_fontsize = font_size

    # x-margins and dot distance
    if x_margin is None:
        x_margin = 0.5 * cloud_width
    x_start = 1.0
    if dot_distance is None:
        dot_distance = cloud_width / 10.0 

    # Means per condition
    means = np.array([np.nanmean(d) for d in data_list])

    # CI parameters
    confidence_interval = 0.95

    # Colors for lines
    alpha_lines = connect_alpha
    color_dominant = (1.0, 0.5529, 0.3608, alpha_lines)   # orange
    color_non_dominant = (0.4, 0.4, 0.4, alpha_lines)      # gray
    color_all = (0.6, 0.6, 0.6, alpha_lines)               # uniform gray

    # ---- create fig/axes and set fonts ----
    with mpl.rc_context(rc={'font.family': font_name, 'font.size': font_size}):
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        mpl.rcParams["text.usetex"] = False

        if ax is None:
            #fig, ax = plt.subplots(figsize=figure_size)
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.set_title(title, fontsize=font_size, fontname=font_name)
        ax.set_xlabel(label_x, fontsize=font_size, fontname=font_name)
        ax.set_ylabel(label_y, fontsize=font_size, fontname=font_name)

        # Midline
        if reference_value is not None:
            ax.axhline(
                reference_value,
                linestyle='--',
                linewidth=line_width*0.8,
                color="grey",
                alpha=0.5,
                zorder=0
            )

        # Prepare storage for x positions
        xRefs_log = []
        dot_anchor_x = [None, None]

        pair_distance_half = pair_gap / 2.0

        # ---- loop over the two conditions ----
        for n in range(N_conditions):

            # position + orientation for this condition
            # left condition (n=0) at x_start - pair_gap/2, bulging to the right
            # right condition (n=1) at x_start + pair_gap/2, bulging to the left
            side = -1 if n == 0 else 1
            cloud_orientation = side    # +1 for left cond, -1 for right cond
            xRef = x_start + side * pair_distance_half

            xRefs_log.append(xRef)

            # draw one semi-violin + dots + mean/SEM/CI
            semi_violin_dict = _draw_one_semi_violin(
                ax,
                data=data_list[n],
                xRef=xRef,
                color=my_colors[n],
                dot_color=dot_colors[n],
                cloud_width=cloud_width,
                cloud_orientation=cloud_orientation,
                dot_distance=dot_distance,
                dot_size=dot_size,
                dot_alpha=dot_alpha,
                line_width=line_width,
                sem_cap_width=sem_cap_width,
            )
            # store anchor for connecting lines later
            dot_anchor_x[n] = semi_violin_dict["anchor_x"]


        # ---- y-limits ----
        if y_limits is not None:
            ax.set_ylim(y_limits[0], y_limits[1])
        else:
            ymin = np.nanmin(all_valid)
            ymax = np.nanmax(all_valid)
            if ymin == ymax:
                margin = 0.5 if ymin == 0 else 0.5 * abs(ymin)
            else:
                margin = 0.05 * (ymax - ymin)
            ax.set_ylim(ymin - margin, ymax + margin)

        # y-ticks
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(
                [str(t) for t in y_ticks],
                fontfamily=font_name,
                fontsize=font_size,
            )

        # ---- x-limits and ticks ----
        xmin = min(xRefs_log) - semi_violin_dict['rect_width'] - x_margin
        xmax = max(xRefs_log) + semi_violin_dict['rect_width'] + x_margin
        ax.set_xlim(xmin, xmax)
        ax.set_xticks(xRefs_log)

        if x_tick_labels is not None and len(x_tick_labels) == 2:
            ax.set_xticklabels(x_tick_labels)
        else:
            ax.set_xticklabels([])
        

        # ---- connecting lines between the two conditions ----
        x_left, x_right = dot_anchor_x[0], dot_anchor_x[1]
        a_clean, b_clean = _paired_clean(a, b)
        for ai, bi in zip(a_clean, b_clean):
            if highlight_connected_lines_in_dominant_direction and not (
                np.isnan(means[0]) or np.isnan(means[1])
            ):
                col = (
                    color_dominant
                    if np.sign(ai - bi) == np.sign(means[0] - means[1])
                    else color_non_dominant
                )
            else:
                col = color_all
            ax.plot(
                [x_left, x_right],
                [ai, bi],
                color=col,
                lw=connect_linewidth,
                zorder=2,
            )

        # ---- significance stars (paired t-test A vs B) ----
        if show_significance_stars:
            from matplotlib.transforms import blended_transform_factory

            if a_clean.size > 1:
                _, p = stats.ttest_rel(a_clean, b_clean, nan_policy="omit")
            else:
                p = np.nan
            stars = significance_stars(p)

            x_bar = [xRefs_log[0], xRefs_log[1]]
            x_stars = 0.5 * (x_bar[0] + x_bar[1])

            trans = blended_transform_factory(ax.transData, ax.transAxes)
            y_bar_axes = 1.02  # slightly above top of axes

            # horizontal bar
            ax.plot(
                x_bar,
                [y_bar_axes, y_bar_axes],
                color=(0, 0, 0, 1),
                lw=0.5,
                transform=trans,
                clip_on=False,
                zorder=6,
            )

            # text (stars or n.s.)
            fsize = star_fontsize if stars in ("*", "**", "***") else ns_fontsize
            y_text_axes = y_bar_axes + (0.01 if stars == "n.s" else -0.04)
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
                zorder=7,
            )

        # ---- styling ----
        # set height of axes
        set_axes_size_inches(ax, target_width_in=figure_size[0], target_height_in=figure_size[1])
        # set axis lines connecting the tick marks
        axis_linewidth = 0.5
        ax.grid(False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color("black")
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(axis_linewidth)
        # make tick marks turn inwards/outwards and set thickness
        ax.tick_params(
            axis="both",
            which="major",
            length=4,
            width=axis_linewidth,
            direction="in",
            labelsize=font_size,
        )
        # impose font name and size for tick labels
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontname(font_name)
            lbl.set_fontsize(font_size)

        return fig, ax
