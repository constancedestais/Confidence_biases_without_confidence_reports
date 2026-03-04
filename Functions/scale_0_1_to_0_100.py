def scale_0_1_to_0_100(series, label: str):
    """Before scaling, assert series is in [0,1] or [-1,1]. Return scaled series."""
    assert series.min() >= -1 and series.max() <= 1, f"Error: {label} should be in [0,1] or [-1,1] before scaling, but min={series.min()} and max={series.max()}"
    return series * 100