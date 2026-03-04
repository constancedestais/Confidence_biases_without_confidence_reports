import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
from scipy import stats
from Functions.set_axes_size import set_axes_size_inches
from Functions.raincloud_helpers import _kde_on_support, _paired_clean, _draw_one_semi_violin
from Functions.significance_stars import significance_stars


def raincloud_unpaired(
    data,
    *,
    my_colors: tuple[list[float]] = ((0, 0, 0),), # must have a comma to make sure it is a tuple
     y_limits: list[float] = None,
    ax=None,
    reference_value: float = 0.0,
    # axis labels & ticks
    title: str = "",
    label_x: str = "",
    label_y: str = "",
    x_tick_labels=None,   # list of labels, length == n_conditions or None
    y_ticks=None,
    y_tick_labels=None,
    # figure + fonts
    figure_size=(6, 4), # width, height of axis, in inches
    font_size: float = 10,
    font_name: str = "Arial",
    # geometry / layout
    cloud_width: float = 0.75,
    x_margin: float | None = None,
    condition_gap: float = 1.0,       # spacing between conditions on x-axis
    # dots
    dot_size: int = 30,
    dot_color=None,
    dot_alpha: float = 0.5,
    dot_distance: float | None = None,  # lateral offset of dots from violin axis
    # mean / error bars
    line_width: float = 1.0,
    sem_cap_width: float = 5.0,
    # significance stars
    show_significance_stars: bool = False,
    star_fontsize: float | None = None,
    ns_fontsize: float | None = None,
):
    """
    Raincloud-style plot for one or more UNPAIRED conditions.

    Each condition is shown as a one-sided (half) violin,
    all facing the SAME direction (to the right).
    Individual points are displayed (no connecting lines).

    If reference_value is provided and show_significance_stars=True,
    a one-sample t-test is performed for each condition versus
    reference_value, and significance stars are drawn above each
    condition.
    """

    # ---- normalize data to a list of 1D arrays ----
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data_list = [data.astype(float).ravel()]
        elif data.ndim == 2:
            # interpret rows as conditions
            data_list = [np.asarray(row, dtype=float).ravel() for row in data]
        else:
            raise ValueError("data array must be 1-D or 2-D.")
    elif isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise ValueError("data is empty.")
        first = data[0]
        # If first element is scalar-like, treat as a single condition
        if not isinstance(first, (list, tuple, np.ndarray)):
            data_list = [np.asarray(data, dtype=float).ravel()]
        else:
            data_list = [np.asarray(d, dtype=float).ravel() for d in data]
    else:
        raise ValueError("Unsupported data type for 'data'.")

    N_conditions = len(data_list)

    # Remove NaNs only for global stats (y-limits)
    all_valid = np.concatenate([d[~np.isnan(d)] for d in data_list]) if N_conditions > 0 else np.array([])
    if all_valid.size == 0:
        raise ValueError("All conditions are empty or all NaN.")

    # Dot colors: single or per-condition
    if dot_color is None:
        dot_colors = my_colors
    else:
        if (
            isinstance(dot_color, (list, tuple))
            and len(dot_color) == N_conditions
            and isinstance(dot_color[0], (list, tuple, np.ndarray))
        ):
            dot_colors = list(dot_color)
        else:
            dot_colors = [dot_color] * N_conditions

    # ---- geometry parameters ----
    if x_margin is None:
        x_margin = 0.5 * cloud_width
    x_start = 1.0
    if dot_distance is None:
        dot_distance = cloud_width / 10.0

    cloud_orientation = 1.0  # all violins bulge to the RIGHT

    # CI parameters
    confidence_interval = 0.95

    # Star font sizes
    if star_fontsize is None:
        star_fontsize = font_size + 1
    if ns_fontsize is None:
        ns_fontsize = font_size

    # ---- create fig/axes and set fonts ----
    with mpl.rc_context(rc={'font.family': font_name, 'font.size': font_size}):

        # enusre that text in SVG output remains text, not paths, for compatibility with Inkscape
        mpl.rcParams["svg.fonttype"] = "none"        # <-- keep text as text
        mpl.rcParams["text.usetex"] = False          # TeX often forces path-like output


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
                zorder=0,
            )

        xRefs_log = []
        condition_pvals = []  # store p-values if we do tests
        cloud_orientation = 1.0  # all violins bulge to the RIGHT

        # ---- loop over conditions ----
        for n, arr in enumerate(data_list):

            # Position of this condition on the x-axis
            xRef = x_start + n * condition_gap
            xRefs_log.append(xRef)

            # Draw one semi-violin + dots + mean/SEM/CI
            semi_violin_dict = _draw_one_semi_violin(
                ax,
                data=arr,
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
            

            # ---- One-sample t-test vs reference_value (for significance stars) ----
            if reference_value is not None and semi_violin_dict["Nsub"] > 1:
                # Use original array arr, not NaN-stripped copy from helper, or: data_clean = arr[~np.isnan(arr)]
                data_clean = np.asarray(arr, dtype=float)
                data_clean = data_clean[~np.isnan(data_clean)]
                if data_clean.size > 1:
                    _, my_p_value = stats.ttest_1samp(
                        data_clean,
                        popmean=reference_value,
                        nan_policy="omit",
                    )
                else:
                    my_p_value = np.nan
            else:
                my_p_value = np.nan
            condition_pvals.append(my_p_value)

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
            # add tick labels provided
            if y_tick_labels is not None and len(y_tick_labels) == len(ax.get_yticks()):
                ax.set_yticklabels(
                    y_tick_labels,
                    fontfamily=font_name,
                    fontsize=font_size,
                )
            else: # generate string labels from tick values
                ax.set_yticklabels(
                    [str(t) for t in y_ticks],
                    fontfamily=font_name,
                    fontsize=font_size,
                )
            


        # ---- x-limits and ticks ----
        xmin = min(xRefs_log) - dot_distance - x_margin
        xmax = max(xRefs_log) + semi_violin_dict['rect_width'] + x_margin
        ax.set_xlim(xmin, xmax)

        ax.set_xticks(xRefs_log)
        if x_tick_labels is not None and len(x_tick_labels) == N_conditions:
            ax.set_xticklabels(x_tick_labels)
        else:
            ax.set_xticklabels([])

        # ---- significance stars per condition (one-sample vs. midline) ----
        if show_significance_stars and reference_value is not None:
            from matplotlib.transforms import blended_transform_factory

            trans = blended_transform_factory(ax.transData, ax.transAxes)
            # y in axes coords; we go slightly above the top
            y_bar_axes = 1.02

            for xRef, my_p_value in zip(xRefs_log, condition_pvals):
                stars = significance_stars(my_p_value)
                # small bar centered at xRef
                bar_half_width = cloud_width * 0.15
                x_bar = [xRef - bar_half_width, xRef + bar_half_width]
                x_stars = xRef

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
                # tiny vertical adjustment for "n.s." if you like
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
