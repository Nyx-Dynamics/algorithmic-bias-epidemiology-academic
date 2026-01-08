# Figure Captions - Academic Publication

## Algorithmic Bias Epidemiology Framework

**For Peer-Reviewed Publication**

---

### Figure 1: individual_barrier_effects.png

**Title:** Marginal Effects of Individual Barrier Removal in the 11-Barrier Algorithmic Discrimination Model

**Caption:** Counterfactual analysis of individual barrier removal effects on success probability. Horizontal bars represent the marginal change in P(success) when each barrier is removed while all others remain in place. Barriers are stratified by layer: Data Integration (green, n=3), Data Accuracy (blue, n=3), and Institutional (red, n=5). All individual effects approach zero (range: 0.00-0.02%), demonstrating the multiplicative blocking structure of the barrier system. This result is consistent with a synergistic interaction model where P(success) = ∏ᵢ P(pass barrier ᵢ), and removal of any single barrier yields P(success|barrier ᵢ removed) ≈ ∏ⱼ≠ᵢ P(pass barrier ⱼ) ≈ 0 when remaining barriers have low pass probabilities. Shapley value decomposition (see Figure 5) provides appropriate attribution accounting for interaction effects.

---

### Figure 2: stepwise_comparison.png

**Title:** Success Probability Trajectories Under Five Stepwise Barrier Removal Strategies

**Caption:** Comparative analysis of barrier removal ordering strategies. X-axis: cumulative number of barriers removed (0-11). Y-axis: success probability (0-100%). Strategies: Forward removal by layer (L1→L2→L3, green circles), Backward removal (L3→L2→L1, red circles), Greedy by marginal impact (purple circles), Greedy by cost-effectiveness (orange circles), and Random ordering (gray circles, mean of n=10 permutations). All strategies exhibit characteristic "hockey stick" trajectories with near-zero improvement until removal of final 2-3 barriers, followed by rapid increase to ~95% success. Strategy equivalence (ANOVA: F=0.23, p=0.92) confirms that removal ordering is irrelevant; only removal completeness determines outcome. This finding has significant policy implications: partial reform efforts addressing subsets of barriers will fail regardless of which barriers are targeted.

---

### Figure 3: layer_effects.png

**Title:** Barrier Layer Removal Effects and Cost-Effectiveness Analysis

**Caption:** **Panel A:** Success probability improvement (Δ%) from removing barrier layers individually and in combination. Single-layer removal: Data Integration (+0.0%), Data Accuracy (+0.0%), Institutional (+0.3%). Dual-layer combinations: +3.2% to +7.8%. Complete removal (all three layers): +95.0%. The nonlinear scaling demonstrates strong positive synergy between layers. **Panel B:** Cost-effectiveness scatter plot. X-axis: total intervention cost (USD). Y-axis: success probability improvement (%). Point size proportional to number of barriers removed. The relationship exhibits increasing returns to scale: marginal cost-effectiveness improves with intervention scope. Complete barrier removal (~$14,000) achieves 6.8% improvement per $1,000, compared to <0.1% per $1,000 for partial interventions. Economic analysis supports comprehensive over incremental reform strategies.

---

### Figure 4: interaction_heatmap.png

**Title:** ANOVA Decomposition of Layer Interaction Effects

**Caption:** Heatmap representation of main effects and two-way interactions from 2³ factorial analysis of layer removal. Diagonal cells: main effects of individual layer removal (Data Integration: 0.0%, Data Accuracy: 0.0%, Institutional: 0.3%). Off-diagonal cells: two-way interaction terms representing deviation from additivity. Color scale: red (negative/antagonistic) to green (positive/synergistic). All pairwise interactions are positive, indicating synergistic relationships. The Data Integration × Institutional interaction (+7.6%) is largest, reflecting the interdependence between data propagation systems and institutional remediation barriers. **Not shown:** Three-way interaction term = +87.6%, accounting for the majority of total effect variance. This decomposition confirms that the barrier system operates as an integrated whole rather than independent components, explaining observed resistance to single-target policy interventions.

---

### Figure 5: shapley_attribution.png

**Title:** Shapley Value Attribution for Barrier Contribution to System Effect

**Caption:** Fair attribution of total success improvement to individual barriers using Shapley value decomposition from cooperative game theory. Shapley values computed over n=1,000 sampled permutations of barrier removal orderings. Bars represent each barrier's average marginal contribution across all possible coalitions. Color coding by layer: Data Integration (green), Data Accuracy (blue), Institutional (red). **Top three attributions:** Legal Knowledge Gap (11.5%), Rapid Data Transmission (10.6%), Systemic Bias in Algorithms (10.3%). Unlike marginal effects (Figure 1), Shapley values appropriately account for interaction structure, identifying barriers whose removal contributes most to overall system improvement. These attributions may inform resource allocation when comprehensive reform is infeasible, though the dominance of the three-way interaction term (87.6%) suggests that partial approaches remain substantially suboptimal.

---

### Figure 6: sensitivity_analysis.png

**Title:** Global Sensitivity Analysis and Model Validation

**Caption:** Four-panel sensitivity analysis of the 11-barrier algorithmic discrimination model. **Panel A (OAT Sensitivity):** One-at-a-time normalized sensitivity indices showing uniform sensitivity across barriers, confirming no single barrier dominates model output. **Panel B (Sobol Indices):** Variance-based global sensitivity indices with first-order (S₁) and total-order (S_T) effects. First-order indices range 0.04-0.10, while total-order indices cluster around 0.11, indicating substantial interaction effects. The gap between S₁ and S_T quantifies each barrier's participation in higher-order interactions. **Panel C (Morris Screening):** Elementary effects analysis plotting mean absolute effect (μ*) against standard deviation (σ). Barriers cluster in the high-σ region, confirming non-linear effects and interactions. **Panel D (Bootstrap CIs):** Distribution of 1,000 bootstrap samples for baseline success probability with 95% confidence interval (0.0013%-0.0025%). The tight distribution confirms model stability and reproducibility.

---

### Figure 7: snr_robustness.png

**Title:** Signal-to-Noise Ratio Analysis and Key Finding Robustness

**Caption:** Four-panel robustness analysis under parameter uncertainty. **Panel A (SNR Curve):** Signal-to-noise ratio (dB) as a function of multiplicative noise injection (1%-30%). SNR remains positive (>0 dB) until ~25% noise, indicating model conclusions are robust to moderate parameter uncertainty. **Panel B (Three-Way Interaction):** Bootstrap distribution of three-way interaction dominance, showing mean of 99.6% with 100% of samples exceeding the 70% threshold. This confirms the central finding that barrier synergy explains >87% of total effect is robust under resampling. **Panel C (Individual Effects):** Bootstrap distribution of maximum individual barrier effect, confirming all samples show <1% individual effect, validating the finding that single-barrier interventions are ineffective. **Panel D (Barriers Required):** Distribution of barriers needed for 90% success, showing 100% of bootstrap samples require ≥10 barriers, confirming the comprehensiveness requirement is robust. All key findings demonstrate 100% robustness across 1,000 bootstrap iterations.

---

## Methodological Notes

### Model Parameters
- **Baseline success probability:** 0.0018% (product of 11 barrier pass probabilities)
- **Maximum success probability:** 100% (all barriers removed)
- **Monte Carlo simulations:** n=1,000 permutations for Shapley value estimation
- **Cost estimates:** Based on legal aid, credit repair, and advocacy service market rates

### Sensitivity Analysis Methods
- **OAT Sensitivity:** One-at-a-time perturbation analysis with ±10% parameter variation
- **Sobol Indices:** Variance-based global sensitivity analysis using Saltelli sampling (n=1,024 base samples)
- **Morris Screening:** Elementary effects method with r=20 trajectories for interaction detection
- **SNR Analysis:** Signal-to-noise ratio computed across noise levels 1%-30% with n=100 replications each
- **Bootstrap Validation:** n=1,000 bootstrap samples with 95% percentile confidence intervals

### Software Environment
- **Python:** 3.10+
- **Core packages:** NumPy, SciPy, Matplotlib, SALib (Sensitivity Analysis Library)
- **Reproducibility:** Random seed fixed at 42 for all stochastic analyses

---

*Nyx Dynamics LLC | January 2026*
