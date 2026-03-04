
import numpy as np
import pandas as pd


def df_to_array_no_conditions(
    df: pd.DataFrame,
    value_column: str,
):
    """
    No-condition helper:
      - returns a single array with all values in `value_column`
      - returns np.nan for the 'order' (as requested)
    """
    if value_column not in df.columns:
        raise KeyError(f"Column '{value_column}' not found in DataFrame.")

    vals = df[value_column].to_numpy(dtype=float)
    return [vals]

def df_to_arrays_unpaired_conditions(
    df: pd.DataFrame,
    condition_column: str,
    value_column: str,
    *,
    condition_order: list | None = None,
    condition_mapping: dict | None = None,
):
    """
    Function transforms a DataFrame into a list of arrays for each condition.
    Advice:
        Use when each condition is standalone and you don’t need to connect subjects across conditions.
        Good for NON-paired plots in raincloud plots.
    
    INPUT:
    - df: pandas DataFrame containing the data
    - condition_column: name of the column containing condition labels
    - value_column: name of the column containing the values to extract
    - condition_order: optional list specifying the order of conditions e.g. ["gain", "loss"] as they appear in condition_column

    OUTPUT:
    - data_list: One float array per condition, can be of different lengths.
    - condition_order: list of condition labels in the order they appear in the data e.g. ["gain", "loss"]

    e.g. function call:
    data_list, order = df_to_conditions_arrays( df, 
                                                condition_column="condition", 
                                                value_column="value",
                                                condition_order=["A","B","C"]  # optional
                                              )
    """
    # basic checks
    for c in (condition_column, value_column):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in DataFrame.")

    d = df[[condition_column, value_column]].copy()

    # ---- mapping (if any) ----
    if condition_mapping is not None:
        unmapped = ~d[condition_column].isin(condition_mapping.keys())
        if unmapped.any():
            bad = pd.unique(d.loc[unmapped, condition_column])
            raise ValueError(f"Unmapped condition values: {bad.tolist()}")
        d["condition_mapped"] = d[condition_column].map(condition_mapping)
        condition_column_name = "condition_mapped"
        # also re-map condition_order (if provided)
        mapped_order = [condition_mapping[c] for c in condition_order] if condition_order is not None else None
    else:
        condition_column_name = condition_column
        mapped_order = condition_order

    # resolve order (if not provided, keep first-appearance order)
    if mapped_order is None:
        mapped_order = list(pd.unique(d[condition_column_name].dropna()))

    # group by mapped condition and collect values
    grouped = d.groupby(condition_column_name, sort=False)[value_column].apply(np.asarray)

    # build one array per condition in mapped_order (empty if not present)
    data_list = [
        np.asarray(grouped.get(c, np.array([])), dtype=float)
        for c in mapped_order
    ]

    return data_list, mapped_order


def df_to_arrays_paired_conditions(
    df: pd.DataFrame,
    condition_column: str,
    value_column: str,
    subject_column: str,
    *,
    condition_order: list | None = None,
    condition_mapping: dict | None = None,
):
    """
    Function transforms a DataFrame into a list of arrays for each condition, taking into account subject alignment.
    Advice:
        Use when the same subjects appear in multiple conditions and you want paired lines or paired stats.
        Good for paired plots in raincloud plots (type 2 or 4).
        Requires: subject_column (to align the rows across conditions)

    INPUT:
    - df: pandas DataFrame containing the data
    - condition_column: name of the column containing condition labels
    - value_column: name of the column containing the values to extract
    - subject_column: name of the column containing subject identifiers
    - condition_order: optional list specifying the order of conditions e.g. ["gain", "loss"] as they appear in condition_column

    OUTPUT:
    - data_list: One float array per condition, all of length n_subjects.
    - condition_order: list of condition labels in the order they appear in the data e.g. ["gain", "loss"]

    e.g. function call:
    data_list, order = df_to_aligned_arrays(df, 
                                            condition_column="condition", 
                                            value_column="value", 
                                            subject_column="subj",
                                            condition_order=["L","R"]  # optional but recommended for stable order
                                            )    
    """
    # basic checks
    for c in (condition_column, value_column, subject_column):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in DataFrame.")

    d = df[[subject_column, condition_column, value_column]].copy()

    # duplicates guard
    dup = d.duplicated(subset=[subject_column, condition_column], keep=False)
    if dup.any():
        offenders = (d.loc[dup, [subject_column, condition_column]]
                      .value_counts()
                      .reset_index(name="count"))
        raise ValueError(
            "Duplicate rows per (subject, condition) detected.\n"
            f"{offenders.to_string(index=False)}"
        )

    # ---- mapping (if any) ----
    if condition_mapping is not None:
        unmapped = ~d[condition_column].isin(condition_mapping.keys())
        if unmapped.any():
            bad = pd.unique(d.loc[unmapped, condition_column])
            raise ValueError(f"Unmapped condition values: {bad.tolist()}")
        d["condition_mapped"] = d[condition_column].map(condition_mapping)
        condition_column_name = "condition_mapped"

        # also re-map condition_order
        if condition_order is not None:
            mapped_order = [condition_mapping[c] for c in condition_order]
    else:
        condition_column_name = condition_column
        mapped_order = condition_order
    
    # resolve order (if not provided, keep first-appearance order)
    if mapped_order is None:
        mapped_order = list(pd.unique(d[condition_column_name].dropna()))

    # ---- pivot (no aggregation) on mapped labels ----
    pivot = d.pivot(index=subject_column, columns=condition_column_name, values=value_column)
    pivot = pivot.reindex(columns=mapped_order)

    data_list = [pivot[c].to_numpy(dtype=float) for c in mapped_order]
    return data_list, mapped_order



def df_to_arrays_paired_conditions_with_subcategories(
    df: pd.DataFrame,
    condition_column: str,
    value_column: str,
    subject_column: str,
    label_col: str, 
    *,
    condition_order: list | None = None,
    condition_mapping: dict | None = None,
):
    """
    Same as df_to_arrays_paired but you also want a category (e.g., group, cohort) that’s constant per subject to drive the “category split” inside each condition’s CI box.
    If want to show subgroup mean/SEM slices inside each condition’s raincloud.
    Requires: subject_column, condition_column and label_col (one label per subject)

    Build (data_2d, labels_1d) for your 'category split' feature,
    assuming ONE row per (subject, condition) and ONE label per subject.

    Returns
    -------
    (data_2d, labels_1d), condition_order
        data_2d shape: (n_conditions, n_subjects)
        labels_1d len: n_subjects (aligned to data_2d columns/subjects)

    e.g.:
    (Data2D_and_labels, order) = df_to_arrays_paired_with_subcategories(  df,
                                                                    condition_column="condition",
                                                                    value_column="value",
                                                                    subject_column="subj",
                                                                    label_col ="group",
                                                                    condition_order=["L","R","L2","R2"]  # for type 4 style ordering
                                                                    )
    """
    
    # basic checks
    for c in (condition_column, value_column, subject_column, label_col):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in DataFrame.")

    d = df[[subject_column, condition_column, value_column, label_col]].copy()

    # duplicates guard (one row per subject×condition)
    dup = d.duplicated(subset=[subject_column, condition_column], keep=False)
    if dup.any():
        offenders = (
            d.loc[dup, [subject_column, condition_column]]
              .value_counts()
              .reset_index(name="count")
        )
        raise ValueError(
            "Duplicate rows per (subject, condition) detected.\n"
            f"{offenders.to_string(index=False)}"
        )

    # ---- mapping (if any) ----
    if condition_mapping is not None:
        unmapped = ~d[condition_column].isin(condition_mapping.keys())
        if unmapped.any():
            bad = pd.unique(d.loc[unmapped, condition_column])
            raise ValueError(f"Unmapped condition values: {bad.tolist()}")
        d["condition_mapped"] = d[condition_column].map(condition_mapping)
        condition_column_name = "condition_mapped"
        # also re-map condition_order (if provided)
        mapped_order = [condition_mapping[c] for c in condition_order] if condition_order is not None else None
    else:
        condition_column_name = condition_column
        mapped_order = condition_order

    # ---- pivot (no aggregation) on mapped labels ----
    pivot = d.pivot(index=subject_column, columns=condition_column_name, values=value_column)

    # resolve order if not given
    if mapped_order is None:
        mapped_order = list(pd.unique(pivot.columns))

    # reindex to requested order
    pivot = pivot.reindex(columns=mapped_order)

    # build data_2d (rows=conditions, cols=subjects)
    data_2d = np.vstack([pivot[c].to_numpy(dtype=float) for c in mapped_order])

    # one label per subject; raise on conflicts
    subj_labels = d[[subject_column, label_col]].dropna()
    counts = subj_labels.groupby(subject_column)[label_col].nunique(dropna=True)
    conflicts = counts[counts > 1]
    if not conflicts.empty:
        raise ValueError(
            "Conflicting labels for some subjects; expected exactly one label per subject.\n"
            f"{conflicts.to_string()}"
        )
    first_labels = (
        subj_labels.drop_duplicates(subset=[subject_column], keep="first")
                   .set_index(subject_column)[label_col]
    )
    labels_1d = first_labels.reindex(pivot.index).to_numpy()

    return (data_2d, labels_1d), mapped_order