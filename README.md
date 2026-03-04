# Confidence Biases Without Confidence Reports

Analysis code for a behavioral study on confidence-related biases without explicit confidence reports, using multiple task datasets and paper-ready figures/statistics.

## Overview

This repository provides scripts to:

- Load task-level CSV datasets (`LearningTask`, `SymbolChoice`, `PairChoice`, `Demographics`, etc.)
- Compute participant-level behavioral summaries (including gain-loss contrasts)
- Run inferential statistics (t-tests, OLS, ANOVAs) for manuscript reporting
- Generate SVG figures for publication

## Repository Structure

- `figures_for_paper.py`  
  Main figure-generation script. Saves outputs to `Outputs/Figures/`.
- `stats_for_paper.py`  
  Main statistics script. Prints manuscript-ready stats to console.
- `Functions/`  
  Reusable helper modules for preprocessing, plotting, and statistics.
- `Data/`  
  Input CSV files (local only; git-ignored).
- `Outputs/`  
  Generated figures and text outputs (local only; git-ignored).

## Requirements

Use Python **3.10+** (the code uses `match/case`).

Python packages used in the project:

- `numpy`
- `pandas`
- `scipy`
- `statsmodels`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `patsy`

## Setup

Create and activate a virtual environment, install dependencies, and create output folders:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy statsmodels matplotlib seaborn scikit-learn patsy
mkdir -p Outputs/Figures
```

## Required Data Files
Data files will be made available online.
Place these files in the Data/ directory:
- CD1_LearningTask.csv
- CD1_PairChoice.csv
- CD1_SymbolChoice.csv
- CD1_BonusRound.csv
- CD1_Demographics.csv
- CD1_General.csv

## Run Analyses
1) Statistics for manuscript text
- stats_for_paper.py

2) Figures for manuscript
- figures_for_paper.py
Most figure outputs are saved as .svg files in Outputs/Figures/.

## Experiment Version Groups Used in Code
The function filter_experiment_version(...) supports these version filters:
- all
- cd1_2025_click_desired_1_identify_best_1
- cd1_2025_click_desired_0_identify_best_1
- cd1_2025_click_desired_1_identify_best_0
- cd1_2025_click_desired_0_identify_best_0
- cd1_2025_click_desired_1_identify_best_1_difficulty_0_70_0_80
- cd1_2025_click_desired_1_identify_best_0_difficulty_0_70_0_80
- versions_equal_difficulty_across_gain_loss
- versions_asymmetric_difficulty_across_gain_loss
- versions_equal_and_asymmetric_difficulty_click_desired_1

## Citation
If you use this code, please cite the associated paper/preprint (add citation details here).
