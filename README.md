# Structural limits of single-barrier reform in algorithmic recourse: a formal series-system model with implications for digital health

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A formal, cross-domain series-system model of algorithmic recourse**

## Overview

Algorithmic decision systems mediate access to healthcare, credit, employment and housing, and individuals who receive adverse decisions must clear multiple sequential barriers to obtain recourse. We develop a **formal series-system model** of recourse with 11 stages across three layers (data integration, data accuracy, institutional access), parameterized provisionally from cross-domain federal datasets and one healthcare audit, and analyze its structure. Numerical results below are **properties of the model**, not empirical measurements of real recourse.

> **Manuscript status:** Under revision at *PLOS Digital Health* (PDIG-D-26-00342). Preprint on medRxiv (doi:10.64898/2026.02.22.26346836). Archived: Zenodo (DOI:10.5281/zenodo.22312977, v2.0.0).

### Key findings (model-derived unless noted)

- **Theorem (Proposition 1):** single-barrier improvement is algebraically bounded by baseline success — structural, not a simulation result.
- **Baseline success probability:** 0.0018% (model-derived, under the calibrated parameters).
- **Individual barrier removal:** all effects <0.02%; maximum 0.0054% (Legal Knowledge Gap).
- **Three-way interaction share:** 87.6% of achievable improvement — a baseline-relative factorial share (not a variance decomposition); **parameterization- and topology-dependent**, not universal.
- **Strategy comparison (descriptive):** all removal strategies converge only at near-complete removal; ordering is irrelevant to the threshold behavior.
- **Bounded stability:** findings are stable within the prespecified ±10–33% perturbation ranges; the coefficient of variation rises to ~110% at 25% noise, so robustness is **not** claimed under arbitrary uncertainty.
- **Alternative topologies:** the comparative conclusion (single-layer ≪ coordinated) survives moderate cross-barrier correlation (ρ≤0.5) but attenuates under repeated-attempt recourse — delimiting the model's domain of validity.

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

**Proposition 2** (Higher-order interaction under probability-scale multiplicativity): On the probability scale, factorial contrasts of P generate nonzero higher-order (including three-way) interaction terms, whereas on the log scale barriers contribute additively and interactions vanish. The **existence** of higher-order interaction is structural; its **magnitude** (e.g., the 87.6% three-way share) is parameterization- and topology-dependent, not a universal property.

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

make all        # full pipeline: baseline, sensitivity, copula, systemic-bias, derivation, tables, figures, tests, verify
make verify     # claims-vs-code checks + determinism (seed 42)
```

Individual analyses are available as targets (`make baseline`, `make copula`, `make systemic_bias_sensitivity`, `make figures`) and as scripts under `analysis/`. Sobol/Morris indices use custom Monte Carlo A/B (Saltelli/Jansen) estimators cross-checked against SALib.

**Environment:** Python 3.10+, NumPy, SciPy, Matplotlib.
**Random seed:** 42 (fixed; regression tests pin the deterministic headline values).

### Dependencies

- Python ≥ 3.10
- NumPy ≥ 1.21
- SciPy ≥ 1.7
- Matplotlib ≥ 3.5
- SALib ≥ 1.4
- Pandas ≥ 1.3

## Preprint & Submission

- **medRxiv preprint:** doi:10.64898/2026.02.22.26346836
- **Journal:** *PLOS Digital Health* (PDIG-D-26-00342, under revision)

## Citation

### Paper

```bibtex
@article{demidont2026synergistic,
  title={Synergistic barriers to algorithmic recourse in healthcare 
         and administrative systems},
  author={Demidont, A.C.},
  journal={medRxiv},
  year={2026},
  doi={10.64898/2026.02.22.26346836},
  note={Under revision at PLOS Digital Health}
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
  doi = {10.5281/zenodo.22312977},
  url = {https://doi.org/10.5281/zenodo.22312977}
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

Code in this repository is released under the [MIT License](LICENSE). The associated manuscript and data are made available under CC BY 4.0 in accordance with PLOS policy.

## Author

**A.C. Demidont, DO**
Nyx Dynamics, LLC
Email: acdemidont@nyxdynamics.org
ORCID: [0000-0002-9216-8569](https://orcid.org/0000-0002-9216-8569)

---

*This research was conducted independently. The author reports prior employment with Gilead Sciences, Inc. (2020–2024); Gilead had no role in this research.*
