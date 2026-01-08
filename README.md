# Algorithmic Bias as Epidemiological Phenomenon

**A Mathematical Framework for Understanding Algorithmic Discrimination Through the Lens of Infectious Disease Dynamics**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Abstract

We present a formal mathematical framework demonstrating that algorithmic discrimination exhibits epidemiological dynamics isomorphic to viral infection. Using the HIV integration paradigm as our primary model, we prove that: (1) algorithmic scoring functions are primitive recursive and thus exhibit predictable, bounded behavior; (2) feedback loops in scoring systems create self-reinforcing degradation trajectories; (3) data integration across systems creates irreversible harm analogous to proviral integration; and (4) barrier systems exhibit strong synergistic interaction (94.7% three-way effect), explaining the failure of piecemeal reform efforts.

We develop the **Life Success Prevention Theorem**, which states that for any exposure *e* > 0 to algorithmic bias, the steady-state life success score *S*∞ < *S*₀, representing permanent penalty. The only path to *S*∞ = *S*₀ requires intervention before data integration—precisely paralleling HIV prevention dynamics. Population Attributable Fraction (PAF) analysis reveals that 16.7-73.6% of adverse outcomes in affected populations are directly attributable to algorithmic factors, with justice-involved individuals showing the highest attributable fraction.

**Keywords:** algorithmic discrimination, computational epidemiology, feedback dynamics, primitive recursive functions, barrier analysis, health disparities

---

## Repository Structure

```
algorithmic-bias-epidemiology-academic/
├── manuscript/
│   ├── main.tex                    # LaTeX manuscript
│   ├── figures/                    # Publication-quality figures
│   ├── tables/                     # Data tables
│   └── supplementary/              # Extended methods, proofs
├── analysis/
│   ├── life_success_theorem.py     # Core mathematical model
│   ├── barrier_analysis.py         # Counterfactual analysis
│   ├── population_analysis.py      # PAF calculations
│   ├── monte_carlo.py              # Simulation framework
│   └── requirements.txt
├── data/
│   ├── simulation_results/         # Output from analyses
│   └── processed/                  # Processed datasets
├── reproducibility/
│   ├── run_all.sh                  # Master script
│   ├── environment.yml             # Conda environment
│   └── docker/                     # Container build files
└── README.md
```

## Core Theorems

### Theorem 1: Algorithmic Scoring Functions are Primitive Recursive

All commercial scoring systems (FICO, employment screening, insurance underwriting) are constructed from:
- Zero function Z(n) = 0
- Successor function S(n) = n + 1
- Projection functions P^k_i
- Composition and primitive recursion

**Implication:** These functions always terminate and produce deterministic outputs. The output is completely determined by the input—there is no randomness or external agency.

### Theorem 2: Feedback Loops are Mathematically Inevitable

For a scoring function f: State → Score where low scores increase rejection probability:

Let {s_n} be the score sequence where s_{n+1} = f(state_n, rejection_n)

If rejection_n = 1 when s_n < threshold, then:
- {s_n} is monotonically decreasing
- {s_n} is bounded below by 0
- ∴ lim_{n→∞} s_n exists and creates an absorbing state

### Theorem 3: Data Integration Creates Irreversibility

For k independent systems each with correction probability p:

P(complete_correction) = p^k

For p = 0.75, k = 20 systems:
P(correction) = 0.75^20 ≈ 0.003 (0.3%)

After integration across systems, Opportunity < 100% **forever**.

### Theorem 4: Barrier Systems Exhibit Synergistic Interaction

For the 11-barrier multiplicative model:
- Individual barrier effects: ~0%
- Pairwise interactions: 0.2-7.6%
- **Three-way interaction: 94.7%**

This synergy explains why piecemeal reform consistently fails.

---

## The Master Equation

The coupled system of life success domains follows:

$$\frac{dS_i}{dt} = -\alpha_i \cdot e \cdot S_i + \beta_i \cdot (S_{i0} - S_i) + \sum_j A_{ij} \cdot (S_j - S_{j0})$$

Where:
- S_i = Life success in domain i (employment, financial, housing, etc.)
- α_i = Domain susceptibility to algorithmic bias
- e = Exposure level (0 ≤ e ≤ 1)
- β_i = Natural recovery rate
- A_ij = Cross-domain coupling matrix

**Closed-form solution:**

$$S(t) = e^{Mt} \cdot S_0 + M^{-1} \cdot [e^{Mt} - I] \cdot b$$

**Steady state:**

$$S_\infty = -M^{-1} \cdot b$$

**Key result:** For any e > 0, S_∞ < S_0 (permanent penalty guaranteed).

---

## Population Attributable Fraction Analysis

| Population | Prevalence (P_exposed) | Relative Risk (RR) | PAF |
|------------|----------------------|-------------------|-----|
| General Population | 0.40 | 1.5 | 16.7% |
| PWID (People Who Inject Drugs) | 0.85 | 3.2 | 65.2% |
| PWH (People With HIV) | 0.70 | 2.4 | 49.5% |
| Justice-Involved | 0.90 | 4.1 | **73.6%** |

These PAF values indicate the proportion of adverse outcomes that would be eliminated if algorithmic discrimination were removed.

---

## Reproducibility

### Quick Start

```bash
# Clone repository
git clone https://github.com/Nyx-Dynamics/algorithmic-bias-epidemiology-academic.git
cd algorithmic-bias-epidemiology-academic

# Create environment
conda env create -f reproducibility/environment.yml
conda activate algo-epi

# Run all analyses
bash reproducibility/run_all.sh
```

### Docker

```bash
docker build -t algo-epi reproducibility/docker/
docker run -v $(pwd)/output:/app/output algo-epi
```

---

## Target Venues

- **Nature Machine Intelligence** - Algorithmic accountability framework
- **Science / Science Advances** - Policy implications of mathematical findings
- **ACM FAccT** - Fairness, Accountability, Transparency in ML
- **American Journal of Epidemiology** - Surveillance methodology
- **JAMA / Health Affairs** - Health disparities focus

---

## Citation

```bibtex
@article{demidont2026algorithmic,
  title={Algorithmic Bias as Epidemiological Phenomenon:
         A Mathematical Framework for Understanding Discrimination Dynamics},
  author={Demidont, AC},
  journal={[Target Journal]},
  year={2026},
  publisher={Nyx Dynamics LLC}
}
```

---

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Author

**AC Demidont, DO**
Nyx Dynamics LLC
acdemidont@nyxdynamics.org

---

*Generated: January 8, 2026*
