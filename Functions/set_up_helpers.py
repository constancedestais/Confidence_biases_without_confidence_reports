# utils/paths.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    output: Path
    figures: Path
    raw_data: Path

def project_paths_for_main(main_file: Optional[str | Path] = None) -> ProjectPaths:
    """
    Resolve project directories relative to a *runner script* located at the project root.

    Parameters
    ----------
    main_file : str | Path | None
        Usually pass `__file__` from your runner script.
        If None, falls back to sys.argv[0] then CWD (last resort).
    create : bool
        Create output/figure dirs if missing.

    Returns
    -------
    ProjectPaths

    How to use
    -------------
    in main script, at the top
        directories = project_paths_for_main(__file__)
    to read data:
        df = pd.read_csv(directories.data / "LearningTask.csv")
    to save figure:
        fig.savefig(directories.figures / "my_plot.svg")
    """
    import sys

    # Input: you ideally pass __file__ from the runner: P = project_paths_for_main(__file__)
    if main_file is None:
        # Prefer the invoked script; if in an interactive session, fallback to CWD.
        candidate = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path.cwd()
    else:
        candidate = Path(main_file).resolve()

    # Normalize to a folder: If the resolved input is a file (typical), root = that_path.parent; If a directory was passed, root = that_directory.
    root = candidate.parent if candidate.is_file() else candidate

    # name paths for specific directories
    data_dir    = root / "Data"
    output_dir  = root / "Outputs"
    figures_dir = output_dir / "Figures"
    raw_data    = data_dir / "raw_data"
    
    # If these folders are missing, send warning
    missing_dirs = []
    for d in [data_dir, output_dir, figures_dir, raw_data]:
        if not d.exists():
            missing_dirs.append(d)
    if missing_dirs:
        print(f"Warning: The following directories are missing and will be created: {missing_dirs}")

    return ProjectPaths(
        root=root,
        data=data_dir,
        output=output_dir,
        figures=figures_dir,
        raw_data=raw_data,
    )

# ------ function to load data CSVs


from pathlib import Path
from typing import Mapping, Any

import pandas as pd

def load_multiple_csvs(  data_dir: Path,
                csv_filenames: Mapping[str, str],      # accepts any mapping
            ) -> dict[str, pd.DataFrame]:              # promises an actual dict
    
    dfs: dict[str, pd.DataFrame] = {}

    for key, filename in csv_filenames.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for '{key}': {path}")
        dfs[key] = pd.read_csv(path)

    return dfs