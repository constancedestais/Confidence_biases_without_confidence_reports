
import matplotlib.pyplot as plt
import numpy as np

def set_axes_size_inches(ax, target_width_in=None, target_height_in=None):
    """
    Rescale the figure so that the *data area* of `ax` has the requested
    width and/or height in inches.

    - If only target_height_in is given: axis height is set, width unchanged.
    - If only target_width_in is given: axis width is set, height unchanged.
    - If both are given: axis width and height are both rescaled.

    The overall figure size is changed; the axis position (in figure
    coordinates) is kept the same.
    """
    fig = ax.figure

    # We need a draw so that the layout and bbox are up to date
    fig.canvas.draw()

    # Get axis bounding box in inches
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    current_width_in = bbox.width
    current_height_in = bbox.height

    if current_width_in == 0 or current_height_in == 0:
        # Nothing sensible to do
        return

    # Compute scale factors for each dimension
    scale_w = None
    scale_h = None

    if target_width_in is not None:
        scale_w = target_width_in / current_width_in
    if target_height_in is not None:
        scale_h = target_height_in / current_height_in

    # If neither is requested, nothing to do
    if scale_w is None and scale_h is None:
        return

    # Current figure size
    fig_w, fig_h = fig.get_size_inches()

    # New figure size: apply the relevant scales, leave the other dimension as is
    new_fig_w = fig_w * (scale_w if scale_w is not None else 1.0)
    new_fig_h = fig_h * (scale_h if scale_h is not None else 1.0)

    fig.set_size_inches(new_fig_w, new_fig_h, forward=True)

    # Optional: if you're using tight_layout or constrained_layout, you may
    # want to call them again here, or do a second draw:
    # fig.tight_layout()
    # fig.canvas.draw_idle()

