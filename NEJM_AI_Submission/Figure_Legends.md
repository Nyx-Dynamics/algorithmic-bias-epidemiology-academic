# Figure Legends

## Figure 1: Individual Barrier Removal Effects

**File:** individual_barrier_effects.png

Counterfactual analysis showing marginal effect on success probability of removing each barrier while others remain. All effects approach zero (<0.02%), demonstrating the multiplicative blocking structure of the barrier system. Error bars represent 95% confidence intervals from bootstrap resampling (n=1,000).

---

## Figure 2: Strategy Comparison for Barrier Removal

**File:** stepwise_comparison.png

Comparison of five barrier removal strategies: forward (L1→L2→L3), backward (L3→L2→L1), greedy by marginal impact, greedy by cost-effectiveness, and random ordering. All strategies exhibit characteristic "hockey stick" trajectories with near-zero improvement until removal of final 2-3 barriers, followed by rapid increase to approximately 95% success. Strategy equivalence (ANOVA: F=0.23, p=0.92) confirms that removal ordering is irrelevant; only completeness determines outcome.

---

## Figure 3: Layer Interaction Effects

**File:** interaction_heatmap.png

Heatmap showing main effects (diagonal) and pairwise interactions (off-diagonal) for the three barrier layers. The dominant three-way interaction (87.6% of total effect, not shown in heatmap) accounts for the majority of system behavior. Color intensity represents effect magnitude (% improvement in success probability).

---

## Figure 4: Global Sensitivity Analysis

**File:** sensitivity_analysis.png

Four-panel comprehensive sensitivity analysis: (A) One-at-a-time (OAT) normalized sensitivity indices showing uniform sensitivity across all barriers (S_i ≈ 1.0); (B) Sobol first-order (S1) and total-order (ST) indices with 95% confidence intervals, demonstrating interaction involvement through the S1-ST gap; (C) Morris elementary effects screening with mean absolute effect (μ*) vs. standard deviation (σ); (D) Bootstrap confidence intervals for key parameters (n=1,000 samples).

---

## Figure 5: Signal-to-Noise Ratio and Robustness

**File:** snr_robustness.png

(A) Signal-to-noise ratio (SNR) as a function of parameter noise level (1-30%). SNR remains positive (>0 dB) up to approximately 25% noise, indicating robust model conclusions. (B-D) Bootstrap validation showing 100% robustness of key findings: (B) three-way interaction dominance (>70% threshold), (C) individual effects near zero (<1% threshold), (D) barriers required for 90% success (≥10 threshold). All key findings demonstrate 100% robustness across 1,000 bootstrap samples.

---

## Supplementary Figure S1: Shapley Value Attribution

**File:** shapley_attribution.png

Shapley value decomposition assigning fair responsibility to each barrier accounting for all possible coalition orderings. Unlike marginal effects (which approach 0%), Shapley values reveal true causal contribution. Top contributors: Legal Knowledge Gap (11.5%), Rapid Data Transmission (10.6%), Systemic Bias in Algorithms (10.3%).

---

## Supplementary Figure S2: Layer Effects

**File:** layer_effects.png

Visualization of the hierarchical barrier structure showing the three layers (Data Integration, Data Accuracy, Institutional) and their constituent barriers with pass probabilities. Demonstrates the nested, multiplicative nature of the barrier system.
