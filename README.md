# Confidence Biases Without Confidence Reports

Code for statistics and figures in paper and Supplementary Material. Paper DOI: [WILL BE ADDED UPON ACCEPTANCE OF THE PAPER]

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
- `Outputs/`  
  Generated figures and text outputs.
- `Data/`  
  Input CSV files.

## Required Data Files
Data files will be made available online.
Place these files in the Data/ directory:
- LearningTask.csv
- CD1_PairChoice.csv
- CD1_SymbolChoice.csv
- CD1_BonusRound.csv
- CD1_Demographics.csv
- CD1_General.csv

## Requirements

Use Python **3.14.0+**

Python packages used in the project (c.f. requirements.txt)

contourpy==1.3.3
cycler==0.12.1
fonttools==4.61.1
kiwisolver==1.4.9
matplotlib==3.10.8
numpy==2.4.2
packaging==26.0
pandas==2.3.3
patsy==1.0.2
pillow==12.1.1
python-dateutil==2.9.0.post0
pytz==2026.1.post1
scipy==1.17.1
seaborn==0.13.2
six==1.17.0
statsmodels==0.14.6
tzdata==2025.3


## Setup

Create and activate a virtual environment, install dependencies, and create output folders:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

Then add the Data and Outputs folders from figshare (link provided in paper) into the project directory.

## Run Analyses
1) Statistics for manuscript text
- stats_for_paper.py

2) Figures for manuscript
- figures_for_paper.py
Most figure outputs are saved as .svg files in Outputs/Figures/.

## Experiment Version Groups Used in Code
The function filter_experiment_version(...) supports these version filters:
- all
- cd1_2025_click_desired_1_identify_best_1 (RL+/CFC+ in paper)
- cd1_2025_click_desired_0_identify_best_1 (RL-/CFC+ in paper)
- cd1_2025_click_desired_1_identify_best_0 (RL+/CFC- in paper)
- cd1_2025_click_desired_0_identify_best_0 (RL-/CFC- in paper)
- cd1_2025_click_desired_1_identify_best_1_difficulty_0_70_0_80 (asymRL+/CFC+ in paper)
- cd1_2025_click_desired_1_identify_best_0_difficulty_0_70_0_80 (asymRL+/CFC- in paper)
- versions_equal_difficulty_across_gain_loss (RL+/CFC+, RL-/CFC+, RL+/CFC-, RL-/CFC- in paper)
- versions_asymmetric_difficulty_across_gain_loss (asymRL+/CFC+, asymRL+/CFC- in paper)
- versions_equal_and_asymmetric_difficulty_click_desired_1 (asymRL+/CFC+, asymRL+/CFC-, RL+/CFC+, RL+/CFC-,in paper)


## Citation
If you use this code, please cite the associated paper/preprint (add citation details here).
