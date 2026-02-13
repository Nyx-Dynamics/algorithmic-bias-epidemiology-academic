# Figure Legends

## Algorithmic Discrimination as a Synergistic Barrier System: A Quantitative Interaction-Dominant Model

**AC Demidont, DO | Nyx Dynamics LLC**

---

### Figure 1: individual_barrier_effects.png

**Figure 1. Individual Barrier Removal Effects.** Counterfactual analysis showing marginal effect on success probability of removing each barrier while all others remain in place. All effects approach zero (<0.02%), demonstrating the multiplicative blocking structure of the barrier system. Error bars represent 95% confidence intervals from bootstrap resampling (n=1,000). Pass probabilities for each barrier are derived from publicly available federal datasets and peer-reviewed studies (see Table 1 and Supplementary Table S1 for empirical sources and derivation logic).

---

### Figure 2: stepwise_comparison.png

**Figure 2. Strategy Comparison for Barrier Removal.** Comparison of barrier removal strategies: forward (L1→L2→L3), backward (L3→L2→L1), greedy by marginal impact, and random ordering. All strategies exhibit characteristic "hockey stick" trajectories with near-zero improvement until removal of the final 2–3 barriers, followed by rapid increase to 100% success at complete barrier removal. Strategy equivalence (ANOVA: F=0.07, p=0.98) confirms that removal ordering is irrelevant; only completeness determines outcome.

---

### Figure 3: interaction_heatmap.png

**Figure 3. Layer Interaction Effects.** Heatmap showing main effects (diagonal) and pairwise interactions (off-diagonal) for the three barrier layers. The dominant three-way interaction (87.6% of total effect, not shown in heatmap) accounts for the majority of system behaviour. Colour intensity represents effect magnitude (% improvement in success probability). The interaction structure demonstrates that barriers operate synergistically rather than independently.

---

### Figure 4: sensitivity_analysis.png

**Figure 4. Global Sensitivity Analysis.** Four-panel comprehensive sensitivity analysis: (A) One-at-a-time (OAT) sensitivity radar showing standard deviation of output change for each barrier under ±10% perturbation; (B) Sobol first-order (S1) and total-order (ST) indices with 95% confidence intervals, demonstrating interaction involvement through the S1–ST gap; (C) Morris elementary effects screening with mean absolute effect (μ*) vs. standard deviation (σ), where high σ indicates nonlinear or interaction effects; (D) Interaction contribution (ST − S1) for each barrier, quantifying the proportion of each parameter's influence attributable to interactions. Parameter values used in sensitivity analysis are traceable to empirical sources documented in Supplementary Table S1.

---

### Figure 5: snr_robustness.png

**Figure 5. Signal-to-Noise Ratio and Robustness.** (A) Signal-to-noise ratio (SNR) as a function of multiplicative parameter noise (1–30%); SNR remains positive (>0 dB) up to approximately 25% noise. (B) Output uncertainty propagation showing mean success probability with 95% uncertainty interval as parameter noise increases. (C) Robustness of key qualitative findings under 10% parameter uncertainty (bootstrap n=1,000), including three-way interaction dominance and near-zero individual effects. (D) Output variability quantified as coefficient of variation (CV) versus input uncertainty.

---

## Supplementary Figures

### Figure S1: shapley_attribution.png

**Figure S1. Shapley Value Attribution of Barrier Contributions.** Shapley value decomposition assigning fair responsibility to each barrier accounting for all possible coalition orderings. Unlike marginal effects (which approach 0%), Shapley values reveal true causal contribution under interaction-dominant dynamics. Top contributors: Legal Knowledge Gap (11.3%), Rapid Data Transmission (11.1%), Correction Process Barriers (10.7%).

---

### Figure S2: layer_effects.png

**Figure S2. Layer Removal Effects.** Effect of removing individual layers and layer combinations on system success probability. Single-layer removal yields negligible improvement. Even two-layer removal produces only modest effects (Data Accuracy + Institutional: 7.4%). Only complete removal across all three layers yields 100% success, reinforcing the synergistic barrier structure.

---

### Figure S3: snr_robustness.png (same as Figure 5)

**Figure S3. Robustness of Model Findings Under Parameter Uncertainty.** (A) Signal-to-noise ratio (SNR) as a function of multiplicative parameter noise (1–30%); SNR remains positive (>0 dB) up to approximately 25% noise. (B) Output uncertainty propagation showing mean success probability with 95% uncertainty interval as parameter noise increases. (C) Robustness of key qualitative findings under 10% parameter uncertainty (bootstrap n=1,000), including three-way interaction dominance and near-zero individual effects. (D) Output variability quantified as coefficient of variation (CV) versus input uncertainty.

---

## Methodological Notes

- **Baseline success probability:** 0.0018% (product of 11 barrier pass probabilities)
- **Strategies compared:** 4 (forward, backward, greedy by marginal impact, random)
- **Shapley values:** Computed over all possible coalition orderings
- **Sobol indices:** Saltelli sampling (n=1,024 base samples)
- **Morris screening:** r=20 trajectories
- **SNR analysis:** Noise levels 1–30%, n=100 replications each
- **Bootstrap validation:** n=1,000 samples, 95% percentile CIs
- **Random seed:** 42
- **Software:** Python 3.10+, NumPy, SciPy, Matplotlib, SALib

---

*Updated: February 2026*
