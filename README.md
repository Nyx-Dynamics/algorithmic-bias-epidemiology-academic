# Algorithmic Discrimination as a Synergistic Barrier System

**A Quantitative Interaction-Dominant Model**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Abstract

Algorithmic systems increasingly mediate access to employment, housing, credit, and healthcare. Despite regulatory interventions, algorithmic discrimination persists. We hypothesised that barriers to resolving algorithmic discrimination function synergistically, such that incremental reforms are structurally insufficient.

We developed a formal multiplicative barrier model comprising 11 barriers across three interacting layers (data integration, data accuracy, institutional access). Using counterfactual analysis, factorial decomposition, Shapley value attribution, and global sensitivity methods (Sobol and Morris screening), we quantified interaction structure and intervention effects.

Baseline resolution probability was 0.0018%. Removal of any single barrier yielded negligible improvement (<0.02%). Variance decomposition revealed that the dominant three-way interaction among layers accounted for 87.6% of total effect. Sensitivity analyses confirmed interaction-dominant dynamics across parameter uncertainty (bootstrap n=1,000). Strategy simulations demonstrated that barrier removal order is irrelevant (ANOVA: F=0.07, p=0.98); only near-complete removal yields substantial improvement.

**Keywords:** algorithmic discrimination, barrier systems, interaction effects, sensitivity analysis, health-care AI, policy modelling

---

## Repository Structure

```
algorithmic-bias-epidemiology-academic/
├── analysis/
│   ├── barrier_visualization.py    # Counterfactual analysis & figure generation (canonical)
│   ├── sensitivity_analysis.py     # Sobol, Morris, OAT, bootstrap, SNR (canonical)
│   ├── barrier_analysis.py         # Exploratory model (deprecated; earlier parameterisation)
│   ├── population_analysis.py      # Population attributable fraction analysis
│   └── requirements.txt
├── data/
│   ├── processed/                  # Output CSVs from analyses
│   │   ├── barrier_definitions.csv
│   │   ├── individual_barrier_effects.csv
│   │   ├── interaction_effects.csv
│   │   ├── shapley_values.csv
│   │   ├── sobol_indices.csv
│   │   ├── snr_analysis.csv
│   │   └── stepwise_*.csv
│   ├── parameter_sources/
│   │   └── parameter_derivation.md # Full derivation logic for all parameters
│   └── literature_review/
├── manuscript/
│   ├── figures/                    # Publication-quality figures (Figures 1-5, S1-S3)
│   └── FIGURE_CAPTIONS.md          # Corrected figure legends
├── reproducibility/
│   ├── environment.yml
│   └── run_all.sh
└── README.md
```

## Key Findings

1. **Baseline success probability:** 0.0018% -- fewer than 2 in 100,000 individuals successfully navigate all barriers
2. **Individual barrier removal:** All effects <0.02% -- single-target interventions are structurally futile
3. **Three-way interaction dominance:** 87.6% of total effect variance -- barriers operate as a synergistic system
4. **Strategy equivalence:** ANOVA F=0.07, p=0.98 -- removal ordering is irrelevant; only completeness matters
5. **Robustness:** SNR positive up to 25% parameter noise; 100% bootstrap robustness (n=1,000) for all principal findings

## Empirical Parameter Traceability

All 11 barrier pass probabilities are derived from publicly available federal datasets and peer-reviewed studies:

| Source | Barriers Informed |
|--------|-------------------|
| CFPB Consumer Response Reports (2022) | Data Integration (L1), Data Accuracy (L2) |
| FTC Section 319 Reports (2013, 2015) | Data Accuracy (L2), Data Integration (L1) |
| Legal Services Corporation Justice Gap (2022) | Institutional (L3): Awareness, Legal Knowledge, Legal Resources |
| Obermeyer et al. *Science* (2019) | Institutional (L3): Systemic Bias in Algorithms |

Each parameter includes: (i) primary source citation, (ii) empirical statistic used, (iii) mapping rationale, and (iv) plausible uncertainty range. Full derivation logic: `data/parameter_sources/parameter_derivation.md`

## Reproducibility

```bash
# Clone repository
git clone https://github.com/Nyx-Dynamics/algorithmic-bias-epidemiology-academic.git
cd algorithmic-bias-epidemiology-academic

# Install dependencies
pip install numpy scipy matplotlib pandas SALib

# Run canonical analyses
python analysis/barrier_visualization.py    # Figures 1-3, S1-S2, CSV exports
python analysis/sensitivity_analysis.py     # Figures 4-5, S3, sensitivity CSVs
```

**Environment:** Python 3.10+, NumPy, SciPy, Matplotlib, SALib
**Random seed:** 42 (fixed for reproducibility)

## Preprint & Submission

- **bioRxiv:** [forthcoming]
- **Target journal:** PLOS ONE

## Citation

```bibtex
@article{demidont2026algorithmic,
  title={Algorithmic Discrimination as a Synergistic Barrier System:
         A Quantitative Interaction-Dominant Model},
  author={Demidont, AC},
  journal={bioRxiv},
  year={2026},
  doi={10.1101/2026.XX.XX.XXXXXX}
}
```

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Author

**AC Demidont, DO**
Nyx Dynamics LLC
acdemidont@nyxdynamics.org

---

*Updated: February 2026*
