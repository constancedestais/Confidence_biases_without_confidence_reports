import matplotlib

from Functions.significance_stars import significance_stars 
matplotlib.use("Agg")  # or "Qt5Agg", depending on what you have installed
import matplotlib.pyplot as plt
from Functions.set_axes_size import set_axes_size_inches
from Functions.my_colors import load_my_colors
import numpy as np


def custom_barplot(x:list, # n_datasets (number of bars)
                    y:list, # !! ONE VALUE for each dataset (bar height), size: n_datasets x 1
                    y_ci:list[list], # upper and lower 95% CI bounds, for each dataset, size: n_datasets x 2 
                    y_reference_value:float = None, # reference value to plot as dashed line
                    y_label:str = None, # y-axis label
                    p_values:list = None, # p-values for each bar, size: n_datasets x 1
                    y_ticks:list = None, # y-ticks to use
                    y_lim:tuple = (None, None), # y-limits
                    x_tick_labels:list = None, # x-tick labels to use
                    x_lim:tuple = (-0.5, None), # x-limits -> need this reference for bar width variable to make sense
                    error_capsize:float = 5,
                    color:str = 'grey', 
                    width:float = 0.5, 
                    star_fontsize:float = 12, 
                    ns_fontsize:float = 8, 
                    font_name:str = 'Arial', 
                    font_size:int = 10, 
                    axis_linewidth:float = 1, 
                    figure_size:tuple = (2, 2), 
                    filename:str = None,
                ):
    
    # Normalize inputs (accept scalars/lists; ensure numpy shapes)
    y = np.asarray(y)
    y_ci = np.asarray(y_ci)
    if y_ci.ndim == 1:
        y_ci = y_ci.reshape(1, -1)
    p_values = np.asarray(p_values) if p_values is not None else None

    # error bars will represent 95% CI
    # yerr should just be ONE value per bar (the half-length of the CI) 
    y_err = abs(y_ci[:, 0]-y_ci[:, 1])/2

    # plotting
    fig, ax = plt.subplots()
    fig.show()
     
    bars = ax.bar(x, y, color=color, edgecolor=color, width=width)
    ax.errorbar(x, y, yerr=y_err, fmt='k.', capsize=error_capsize, lw=1.2)
    ax.axhline(y_reference_value, color='gray', linestyle='--', lw=1)
    # Add stars above bars (only if p_values provided)
    if p_values is not None:
        offset = 0.03 * (y_ci[:, 1].max() - y_ci[:, 0].min())
        for dataset_d, p_val in enumerate(p_values):
            my_stars = significance_stars(p_val)
            fsize = star_fontsize if my_stars in ("*", "**", "***") else ns_fontsize
            y_text_axes = y_ci[dataset_d, 1] + offset
            ax.text(
                x[dataset_d],
                y_text_axes,
                my_stars,
                fontsize=fsize,
                fontname=font_name,
                ha="center",
                va="bottom",
                clip_on=False,
                zorder=7,
            )

    # axis labels
    ax.set_ylabel(y_label, fontsize=font_size, fontname=font_name)

    # ---- x-limits and ticks ----
    ax.set_xlim(x_lim[0], len(x)-0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(x_tick_labels, fontsize=6, rotation=45, ha='right')

    # ---- y-limits ----
    ax.set_ylim(y_lim[0], y_lim[1])

    # y-ticks
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(
            [str(t) for t in y_ticks],
            fontfamily=font_name,
            fontsize=font_size,
        )

    # ----- styling -----
    # make tick marks turn inwards/outwards and set thickness
    ax.tick_params(
        axis="both",
        which="major",
        length=4,
        width=axis_linewidth,
        direction="in",
        labelsize=font_size,
    )
    # set axis lines connecting the tick marks
    ax.grid(False)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    for spine in ("left",):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("black")
    for axis in ['top','left','right']:
        ax.spines[axis].set_linewidth(axis_linewidth)
    # no tick marks on bottom axis
    ax.xaxis.set_ticks_position('none')
    # impose font name and size for tick labels
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontname(font_name)
        lbl.set_fontsize(font_size)
    # set height of axes
    set_axes_size_inches(ax, target_width_in=figure_size[0], target_height_in=figure_size[1])

    #fig.tight_layout()

    if filename is not None:
        fig.savefig(filename, dpi=300, bbox_inches="tight")

    return fig, ax