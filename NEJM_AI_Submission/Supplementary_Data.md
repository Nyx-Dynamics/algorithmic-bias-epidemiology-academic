# Supplementary Data

## Data Availability

All data files are available in CSV format at:
https://github.com/Nyx-Dynamics/algorithmic-bias-epidemiology-academic/tree/main/data/processed

---

## Table S1: Barrier Definitions and Parameters

**File:** barrier_definitions.csv

| Barrier | Layer | Pass Probability | Estimated Cost ($) | Source |
|---------|-------|-----------------|-------------------|--------|
| Cross-System Data Sharing | Data Integration | 0.30 | 500 | CFPB 2022 |
| Multi-Database Flagging | Data Integration | 0.25 | 300 | Wu & Mayer 2019 |
| Rapid Data Transmission | Data Integration | 0.35 | 200 | Industry estimates |
| Error Correction Difficulty | Data Accuracy | 0.40 | 1,500 | FCRA dispute data |
| Identity Verification | Data Accuracy | 0.45 | 800 | Credit bureau data |
| Systemic Bias in Algorithms | Data Accuracy | 0.35 | 2,000 | Obermeyer 2019 |
| Legal Knowledge Gap | Institutional | 0.20 | 3,000 | LSC 2022 |
| Financial Resources | Institutional | 0.25 | 2,500 | Legal aid data |
| Time Constraints | Institutional | 0.30 | 1,000 | Labor statistics |
| Retaliation Concerns | Institutional | 0.50 | 500 | EEOC data |
| Procedural Complexity | Institutional | 0.35 | 1,200 | Court filing data |

---

## Table S2: Individual Barrier Removal Effects

**File:** individual_barrier_effects.csv

Contains counterfactual analysis results for each barrier showing:
- Baseline success probability
- Success probability with barrier removed
- Marginal effect (Δ)
- 95% confidence intervals

---

## Table S3: Interaction Effects (ANOVA Decomposition)

**File:** interaction_effects.csv

| Effect | Δ Success (%) | % of Total |
|--------|--------------|------------|
| L1 (Data Integration) | 0.0 | 0.0 |
| L2 (Data Accuracy) | 0.0 | 0.0 |
| L3 (Institutional) | 0.3 | 0.3 |
| L1 × L2 | 3.2 | 3.4 |
| L1 × L3 | 7.6 | 8.0 |
| L2 × L3 | 0.5 | 0.5 |
| L1 × L2 × L3 | 83.4 | **87.6** |
| **Total** | 95.0 | 100.0 |

---

## Table S4: Shapley Value Attribution

**File:** shapley_values.csv

| Rank | Barrier | Layer | Shapley Value (%) |
|------|---------|-------|------------------|
| 1 | Legal Knowledge Gap | Institutional | 11.5 |
| 2 | Rapid Data Transmission | Data Integration | 10.6 |
| 3 | Systemic Bias in Algorithms | Data Accuracy | 10.3 |
| 4 | Financial Resources | Institutional | 9.8 |
| 5 | Cross-System Data Sharing | Data Integration | 9.4 |
| 6 | Procedural Complexity | Institutional | 9.2 |
| 7 | Error Correction Difficulty | Data Accuracy | 8.7 |
| 8 | Time Constraints | Institutional | 8.5 |
| 9 | Identity Verification | Data Accuracy | 8.2 |
| 10 | Retaliation Concerns | Institutional | 7.1 |
| 11 | Multi-Database Flagging | Data Integration | 6.7 |

---

## Table S5: Sobol Sensitivity Indices

**File:** sobol_indices.csv

| Barrier | S1 | S1 CI (±) | ST | ST CI (±) |
|---------|-----|----------|-----|----------|
| Cross-System Data Sharing | 0.099 | 0.027 | 0.115 | 0.008 |
| Multi-Database Flagging | 0.065 | 0.027 | 0.112 | 0.007 |
| Rapid Data Transmission | 0.096 | 0.026 | 0.120 | 0.009 |
| Error Correction Difficulty | 0.092 | 0.026 | 0.109 | 0.008 |
| Identity Verification | 0.090 | 0.023 | 0.112 | 0.008 |
| Systemic Bias | 0.088 | 0.024 | 0.114 | 0.007 |
| Legal Knowledge Gap | 0.101 | 0.025 | 0.106 | 0.008 |
| Financial Resources | 0.076 | 0.024 | 0.109 | 0.007 |
| Time Constraints | 0.062 | 0.026 | 0.116 | 0.007 |
| Retaliation Concerns | 0.045 | 0.026 | 0.108 | 0.007 |
| Procedural Complexity | 0.057 | 0.030 | 0.119 | 0.008 |

Note: S1 = first-order index; ST = total-order index; CI = 95% confidence interval

---

## Table S6: Bootstrap Robustness Summary

**File:** robustness_summary.csv

| Finding | Threshold | Bootstrap Mean | Robustness (%) |
|---------|-----------|----------------|----------------|
| Three-way interaction dominance | >70% | 99.6% | 100.0 |
| Individual effects near zero | <1% | 0.0055% | 100.0 |
| Barriers for 90% success | ≥10 | 11.0 | 100.0 |

---

## Table S7: Signal-to-Noise Ratio Analysis

**File:** snr_analysis.csv

| Noise Level (%) | SNR (dB) | Mean Output | Std Output |
|-----------------|----------|-------------|------------|
| 1 | 40.0 | 0.0018 | 0.00002 |
| 5 | 26.0 | 0.0018 | 0.00009 |
| 10 | 20.0 | 0.0018 | 0.00018 |
| 15 | 16.5 | 0.0018 | 0.00027 |
| 20 | 14.0 | 0.0018 | 0.00036 |
| 25 | 1.2 | 0.0018 | 0.00045 |
| 30 | -2.5 | 0.0018 | 0.00054 |

Note: SNR remains positive (>0 dB) up to approximately 25% noise

---

## Code Availability

Analysis code is available at:
https://github.com/Nyx-Dynamics/algorithmic-bias-epidemiology-academic/tree/main/analysis

Key scripts:
- `barrier_analysis.py` - Main barrier model and counterfactual analysis
- `sensitivity_analysis.py` - Sobol, Morris, OAT, and bootstrap analysis
- `barrier_visualization.py` - Figure generation

Environment: Python 3.10+, NumPy, SciPy, Matplotlib, SALib
Random seed: 42 (fixed for reproducibility)
