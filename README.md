# Synergistic Barriers to Algorithmic Recourse in Healthcare and Administrative Systems

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**A Quantitative Interaction-Dominant Model**

## Overview

Algorithmic decision systems mediate access to healthcare, credit, employment and housing, yet individuals who experience adverse decisions face multi-stage barriers when seeking recourse. We formalize these barriers as a series-structured system with 11 empirically parameterized stages across three layers (data integration, data accuracy and institutional access) and prove that single-barrier interventions are bounded by baseline system success.

> **Manuscript status:** Submitted to medRxiv (February 2026). Target journal: *npj Digital Medicine*.

### Key Findings

- **Baseline success probability:** 0.0018% — fewer than 2 in 100,000 individuals successfully navigate all barriers
- **Individual barrier removal:** All effects <0.02% — single-target interventions are structurally futile
- **Three-way interaction dominance:** 87.6% of total effect variance — barriers operate as a synergistic system
- **Strategy equivalence:** ANOVA F=0.07, p=0.98 — removal ordering is irrelevant; only completeness matters
- **Robustness:** SNR positive up to 25% parameter noise; 100% bootstrap robustness (n=1,000) for all principal findings

## Repository Structure

```
algorithmic-bias-epidemiology-academic/
├── analysis/
│   ├── barrier_visualization.py         # Counterfactual analysis & figure generation
│   ├── sensitivity_analysis.py          # Sobol, Morris, OAT, bootstrap, SNR
│   ├── barrier_analysis.py              # Exploratory model (deprecated)
│   ├── population_analysis.py           # Population attributable fraction analysis
│   └── requirements.txt
├── data/
│   ├── processed/                       # Output CSVs from analyses
│   │   ├── barrier_definitions.csv
│   │   ├── individual_barrier_effects.csv
│   │   ├── interaction_effects.csv
│   │   ├── shapley_values.csv
│   │   ├── sobol_indices.csv
│   │   ├── snr_analysis.csv
│   │   └── stepwise_*.csv
│   ├── parameter_sources/
│   │   └── parameter_derivation.md      # Full derivation logic for all parameters
│   └── literature_review/
├── manuscript/
│   ├── medrXiv_manuscript_final.tex     # medRxiv submission manuscript
│   ├── medrXiv_supplement_final.tex     # Supplementary information
│   ├── figures/                         # Publication-quality figures (Figs 1–5, S1–S2)
│   └── FIGURE_CAPTIONS.md
├── reproducibility/
│   ├── environment.yml
│   └── run_all.sh
├── CITATION.cff
├── LICENSE
└── README.md
```

## Mathematical Framework

### Series-Structured Barrier Model

Under the multiplicative barrier model, the probability of successful cascade completion is:

```
P(success) = ∏(i=1 to 11) p_i
```

**Proposition 1** (Single-barrier improvement bound): Removing barrier *j* yields Δ_j = P · (1/p_j − 1). Absolute improvement is linear in baseline success P — an algebraic property of series systems, not a simulation finding.

**Proposition 2** (Interaction dominance on the probability scale): On the probability scale, multiplicative stacking produces threshold behavior and interaction dominance. On the log scale, barriers contribute additively — but policy outcomes and equity metrics are evaluated on the probability scale.

### Three-Layer Framework

| Layer | Barriers | Domain |
|-------|----------|--------|
| L1: Data Integration | Rapid data transmission, multi-system integration, permanent storage | Speed and breadth of adverse data propagation |
| L2: Data Accuracy | Error detection, correction process, incomplete correction propagation | Detecting and correcting erroneous data |
| L3: Institutional | Awareness gap, record access, legal knowledge, legal resources, systemic bias | Awareness, access, legal knowledge and algorithmic bias |

### Empirical Parameter Traceability

All 11 barrier pass probabilities are derived from publicly available federal datasets and peer-reviewed studies:

| Source | Barriers Informed |
|--------|-------------------|
| CFPB Consumer Response Reports (2022) | Data Integration (L1), Data Accuracy (L2) |
| FTC Section 319 Reports (2013, 2015) | Data Accuracy (L2), Data Integration (L1) |
| Legal Services Corporation Justice Gap (2022) | Institutional (L3): Awareness, Legal Knowledge, Legal Resources |
| Obermeyer et al. *Science* (2019) | Institutional (L3): Systemic Bias in Algorithms |

Full derivation logic: `data/parameter_sources/parameter_derivation.md`

## Reproducibility

```bash
git clone https://github.com/Nyx-Dynamics/algorithmic-bias-epidemiology-academic.git
cd algorithmic-bias-epidemiology-academic

pip install numpy scipy matplotlib pandas SALib

# Counterfactual analysis and figures 1–3, S1–S2
python analysis/barrier_visualization.py

# Sensitivity analysis and figures 4–5
python analysis/sensitivity_analysis.py
```

**Environment:** Python 3.10+, NumPy, SciPy, Matplotlib, SALib
**Random seed:** 42 (fixed for reproducibility)

### Dependencies

- Python ≥ 3.10
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- Matplotlib ≥ 3.5
- SALib ≥ 1.4
- Pandas ≥ 1.3

## Preprint & Submission

- **medRxiv:** [DOI pending]
- **Target journal:** *npj Digital Medicine*
- **Previous submissions:** PLOS ONE (PONE-D-26-08047, declined)

## Citation

### Paper

```bibtex
@article{demidont2026synergistic,
  title={Synergistic barriers to algorithmic recourse in healthcare 
         and administrative systems},
  author={Demidont, A.C.},
  journal={medRxiv},
  year={2026},
  doi={PENDING},
  note={Target journal: npj Digital Medicine}
}
```

### Software

```bibtex
@software{demidont2026algorithmic_code,
  author = {Demidont, A.C.},
  title = {algorithmic-bias-epidemiology-academic: Synergistic Barrier 
           Model for Algorithmic Recourse},
  year = {2026},
  publisher = {Zenodo},
  version = {v1.1.0},
  doi = {PENDING},
  url = {https://github.com/Nyx-Dynamics/algorithmic-bias-epidemiology-academic}
}
```

## Related Work

- **Prevention Theorem**: [Nyx-Dynamics/Prevention-Theorem](https://github.com/Nyx-Dynamics/Prevention-Theorem)
- **PWID Structural Barriers**: [Nyx-Dynamics/HIV_Prevention_PWID](https://github.com/Nyx-Dynamics/HIV_Prevention_PWID)
- **LAI-PrEP Bridge Tool**: [Nyx-Dynamics/lai-prep-bridge-tool-pub](https://github.com/Nyx-Dynamics/lai-prep-bridge-tool-pub)
- **Noise Decorrelation in HIV**: [Nyx-Dynamics/noise_decorrelation_hiv](https://github.com/Nyx-Dynamics/noise_decorrelation_hiv)
- **Bridging the Gap — PrEP Cascade**: [Nyx-Dynamics/bridging_the_gap](https://github.com/Nyx-Dynamics/bridging_the_gap)

## Interactive Summary

Explore the full framework — narrated slide deck, infographic, and mind map:
[nyxdynamics.org/research/algorithmic-discrimination](https://nyxdynamics.org/research/algorithmic-discrimination/)

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Author

**A.C. Demidont, DO**
Nyx Dynamics, LLC
Email: acdemidont@nyxdynamics.org
ORCID: [0000-0002-9216-8569](https://orcid.org/0000-0002-9216-8569)

---

*This research was conducted independently. The author reports prior employment with Gilead Sciences, Inc. (2020–2024); Gilead had no role in this research.*
