import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl  
from matplotlib import rcParams
import seaborn as sns
from scipy import stats
import numpy as np
from Functions.set_axes_size import set_axes_size_inches

def create_timeseries_plot( df, 
                            id_column, 
                            condition_column, 
                            trial_column, 
                            value_column,
                            condition_mapping=None, 
                            condition_colors=None,
                            ylabel="", 
                            title="", 
                            reference_value=None,
                            one_indexed=True,                              
                            x_ticks=None, 
                            y_ticks=None,
                            x_limits=None, 
                            y_limits=None, 
                            show_legend=False, 
                            save_path=None,
                            averaging='mean',            # 'mean' or 'median'
                            figure_size=(8, 4),
                            font_size= 16,
                            line_width= 2,
                            line_alpha= 0.9,              # MATLAB lines are prominent
                            ci_alpha= 0.25,               # CI band opacity (FaceAlpha)
                            midline_color= 'black',
                            font_name= 'Arial',
                        ):
    """
    MATLAB-like time series with shaded 95% CI bands for 1–2 conditions.
    """

    # check that main variable is not between 0 and 1 (if it is, likely not scaled to percentage yet)
    assert df[value_column].max() > 1.5, f"Error: {value_column} values look like they are still coded 0-1, please scale to percentage before plotting"

    # sanity checks
    # if there is a condition mapping or condition color input, there should also be a condition column
    assert (condition_column is not None) or (condition_mapping is None and condition_colors is
        None), "If condition_mapping or condition_colors is provided, condition_column must also be provided."
    # if no condition column is provided, condition mapping and colors should also be None
    assert (condition_column is not None) or (condition_mapping is None and condition_colors is
        None), "If condition_column is None, condition_mapping and condition_colors must also be None."
    # df cannot be empty
    assert not df.empty, "Input dataframe is empty."
    # there must be a value column
    assert value_column in df.columns, f"Value column '{value_column}' not found in dataframe."
    # there must be a trial column
    assert trial_column in df.columns, f"Trial column '{trial_column}' not found in dataframe."
    # there must be an id column
    assert id_column in df.columns, f"ID column '{id_column}' not found in dataframe."

    # ---------- Data prep ----------
    # add comments
    # If no condition column is specified or all values are NaN, plot all data together
    if condition_column is None or df[condition_column].isna().all():
        df_plot = df[[id_column, trial_column, value_column]].copy()
        df_plot['__cond__'] = 'All'
        condition_key = '__cond__'
    else: # if there are conditions
        df_plot = df[[id_column, condition_column, trial_column, value_column]].copy()
        # If you provided condition_mapping (e.g., {0: 'Loss', 1: 'Gain'}), code below creates __cond__ holding the mapped labels and sets condition_key to that; otherwise it uses the raw condition column directly. 
        # Why this is useful: the rest of the pipeline only needs to refer to condition_key (never worrying about whether conditions exist, or whether they’re mapped). It also lets color/marker dictionaries be keyed by friendly labels.
        if condition_mapping:
            # Map condition values to labels
            df_plot['__cond__'] = df_plot[condition_column].map(condition_mapping)
            condition_key = '__cond__'
        else:
            condition_key = condition_column

    # Convert to 1-indexed trials if requested rather than 0-indexed
    if one_indexed:
        df_plot[trial_column] = df_plot[trial_column] + 1

    # Get unique conditions
    conditions = list(pd.unique(df_plot[condition_key]))

    # Colors (keep order stable and similar to MATLAB’s explicit color array)
    if condition_colors is None:
        palette = ['black','grey','blue', 'orange', 'green', 'red', 'purple'] #['tab:black','tab:grey','tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        condition_colors = {c: palette[i % len(palette)] for i, c in enumerate(conditions)}


    with mpl.rc_context(rc={'font.family': font_name}):

        # set font settings that should be compatible with Inkscape ("This avoids some of the weird Type 3 font behavior that Inkscape doesn’t always enjoy.")
        mpl.rcParams["pdf.fonttype"] = 42  # TrueType fonts in PDF
        mpl.rcParams["ps.fonttype"] = 42   # same for EPS (if ever)
        mpl.rcParams["text.usetex"] = False

        # ---------- Figure / style ----------
        #fig, ax = plt.subplots(figsize=figure_size)
        fig, ax = plt.subplots()

        n_subj = df_plot[id_column].nunique()

        # ---------- Plot per condition (line + shaded 95% CI) ----------
        for cond in conditions:

            d = df_plot[df_plot[condition_key] == cond]
            grp = d.groupby(trial_column)[value_column]
            if averaging == 'median':
                center = grp.median()
                std = grp.std()
            else:
                center = grp.mean()
                std = grp.std()

            ci = 1.96 * (std / np.sqrt(n_subj))
            x = center.index.values
            y = center.values
            y_lo = (center - ci).values
            y_hi = (center + ci).values

            # CI band (MATLAB fill)
            ax.fill_between(x, y_lo, y_hi,
                            color=condition_colors[cond],
                            alpha=ci_alpha, linewidth=0, zorder=1)

            # Line (MATLAB plot with alpha + linewidth)
            ax.plot(x, y, '-', color=condition_colors[cond],
                    linewidth=line_width, alpha=line_alpha, zorder=2,
                    label=str(cond))

        # ---------- Axes limits / ticks like MATLAB ----------
        if x_limits is not None:
            ax.set_xlim(x_limits)
        else:
            ax.set_xlim(min(df_plot[trial_column]), max(df_plot[trial_column]))

        if y_limits is not None:
            ax.set_ylim(y_limits)

        if x_ticks is not None:
            ax.set_xticks(x_ticks)
            ax.set_yticklabels([str(t) for t in y_ticks],
                            fontfamily=font_name,
                            fontsize=font_size)
            
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(t) for t in y_ticks],
                            fontfamily=font_name,
                            fontsize=font_size)

        # ---------- Labels / title ----------
        #ax.set_xlabel('Trial Number', fontsize=font_size, fontname=font_name)
        ax.set_ylabel(ylabel, fontsize=font_size, fontname=font_name)
        if title:
            ax.set_title(title, fontsize=font_size, fontname=font_name)
        
        # ---------- Legend ----------
        # MATLAB snippet comments out the legend; keep optional
        if show_legend and len(conditions) > 1:
            ax.legend(frameon=False, fontsize=font_size - 2, fontname=font_name)

        # fig.tight_layout()

        # ---------- Midline ----------
        if reference_value is not None:
            ax.axhline(reference_value, linestyle='--', linewidth=line_width*0.8,
                        color="grey", alpha=0.5, zorder=0)
            
        # ---------- styling ----------
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
            
        # ---------- Save like MATLAB print -dsvg ----------
        if save_path:
            # save_path should be the FULL path without extension if you want both;
            # here we mirror MATLAB's SVG export.
            fig.savefig(f"{save_path}.svg", bbox_inches='tight')
            # If you also want a PNG preview, uncomment:
            # fig.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')

        plt.close(fig)
        
        if ax is not None: 
            return ax.figure, ax 
        else: 
            return fig, ax
