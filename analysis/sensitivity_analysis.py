"""
SENSITIVITY ANALYSIS AND MODEL VALIDATION
Algorithmic Bias Epidemiology Framework

Comprehensive robustness testing for peer-reviewed publication including:
1. One-at-a-time (OAT) parameter sensitivity
2. Global sensitivity analysis (Sobol indices)
3. Morris elementary effects screening
4. Signal-to-Noise Ratio (SNR) analysis
5. Bootstrap confidence intervals
6. Monte Carlo uncertainty quantification

Author: AC Demidont, DO
Nyx Dynamics LLC
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# BARRIER MODEL (from barrier_visualization.py)
# =============================================================================

@dataclass
class Barrier:
    """Individual barrier in the algorithmic bias system."""
    name: str
    layer: str
    base_probability: float
    removal_cost: float


class BarrierModel:
    """Simplified barrier model for sensitivity analysis."""

    def __init__(self, barrier_probs: np.ndarray = None):
        """Initialize with 11 barrier probabilities."""
        if barrier_probs is None:
            # Default barrier probabilities
            self.barrier_probs = np.array([
                0.30, 0.55, 0.45,  # Layer 1: Data Integration
                0.35, 0.35, 0.40,  # Layer 2: Data Accuracy
                0.30, 0.55, 0.25, 0.40, 0.30  # Layer 3: Institutional
            ])
        else:
            self.barrier_probs = barrier_probs

        self.n_barriers = len(self.barrier_probs)
        self.layer_indices = {
            'data_integration': [0, 1, 2],
            'data_accuracy': [3, 4, 5],
            'institutional': [6, 7, 8, 9, 10]
        }

    def calculate_success(self, probs: np.ndarray = None) -> float:
        """Calculate success probability (product of barrier probs)."""
        if probs is None:
            probs = self.barrier_probs
        return np.prod(probs)

    def calculate_success_removing(self, indices: List[int]) -> float:
        """Calculate success with specified barriers removed (set to 1.0)."""
        probs = self.barrier_probs.copy()
        for idx in indices:
            probs[idx] = 1.0
        return np.prod(probs)


# =============================================================================
# ONE-AT-A-TIME (OAT) SENSITIVITY ANALYSIS
# =============================================================================

class OATSensitivityAnalysis:
    """
    One-at-a-time sensitivity analysis.

    Varies each parameter individually while holding others constant.
    Measures local sensitivity around baseline values.
    """

    def __init__(self, model: BarrierModel):
        self.model = model
        self.baseline = model.barrier_probs.copy()
        self.baseline_output = model.calculate_success()

    def analyze(self,
                perturbation_range: Tuple[float, float] = (-0.2, 0.2),
                n_points: int = 21) -> pd.DataFrame:
        """
        Perform OAT sensitivity analysis.

        Args:
            perturbation_range: (min, max) relative perturbation
            n_points: Number of perturbation levels to test

        Returns:
            DataFrame with sensitivity results for each parameter
        """
        perturbations = np.linspace(perturbation_range[0], perturbation_range[1], n_points)
        results = []

        for i in range(self.model.n_barriers):
            param_results = []

            for delta in perturbations:
                # Perturb parameter i
                test_probs = self.baseline.copy()
                new_value = self.baseline[i] * (1 + delta)
                new_value = np.clip(new_value, 0.01, 0.99)  # Keep in valid range
                test_probs[i] = new_value

                # Calculate output
                output = self.model.calculate_success(test_probs)

                param_results.append({
                    'barrier_idx': i,
                    'perturbation': delta,
                    'param_value': new_value,
                    'output': output,
                    'output_change': output - self.baseline_output,
                    'output_change_pct': (output - self.baseline_output) / self.baseline_output * 100
                })

            results.extend(param_results)

        return pd.DataFrame(results)

    def calculate_sensitivity_indices(self) -> pd.DataFrame:
        """
        Calculate local sensitivity indices.

        S_i = (∂Y/∂X_i) * (X_i/Y)  [Normalized sensitivity]
        """
        epsilon = 0.01  # Small perturbation
        indices = []

        for i in range(self.model.n_barriers):
            # Forward difference
            probs_plus = self.baseline.copy()
            probs_plus[i] = min(0.99, self.baseline[i] * (1 + epsilon))
            output_plus = self.model.calculate_success(probs_plus)

            # Backward difference
            probs_minus = self.baseline.copy()
            probs_minus[i] = max(0.01, self.baseline[i] * (1 - epsilon))
            output_minus = self.model.calculate_success(probs_minus)

            # Central difference derivative
            dY_dX = (output_plus - output_minus) / (probs_plus[i] - probs_minus[i])

            # Normalized sensitivity
            S_i = dY_dX * (self.baseline[i] / self.baseline_output) if self.baseline_output > 0 else 0

            indices.append({
                'barrier_idx': i,
                'baseline_value': self.baseline[i],
                'dY_dX': dY_dX,
                'normalized_sensitivity': S_i,
                'abs_sensitivity': abs(S_i)
            })

        return pd.DataFrame(indices).sort_values('abs_sensitivity', ascending=False)


# =============================================================================
# GLOBAL SENSITIVITY ANALYSIS (SOBOL INDICES)
# =============================================================================

class SobolSensitivityAnalysis:
    """
    Variance-based global sensitivity analysis using Sobol indices.

    Decomposes output variance into contributions from each input
    and their interactions.

    S_i = V[E(Y|X_i)] / V(Y)  [First-order index]
    S_Ti = 1 - V[E(Y|X_~i)] / V(Y)  [Total-order index]
    """

    def __init__(self, model: BarrierModel, n_samples: int = 10000):
        self.model = model
        self.n_samples = n_samples
        self.baseline = model.barrier_probs.copy()

    def generate_samples(self, bounds_factor: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Sobol sample matrices A and B.

        Uses uniform sampling within bounds around baseline values.
        """
        n = self.model.n_barriers

        # Define bounds for each parameter
        lower = np.maximum(0.05, self.baseline * (1 - bounds_factor))
        upper = np.minimum(0.95, self.baseline * (1 + bounds_factor))

        # Generate two independent sample matrices
        A = np.random.uniform(size=(self.n_samples, n))
        B = np.random.uniform(size=(self.n_samples, n))

        # Scale to parameter bounds
        A = lower + A * (upper - lower)
        B = lower + B * (upper - lower)

        return A, B

    def calculate_indices(self, bounds_factor: float = 0.3) -> pd.DataFrame:
        """
        Calculate first-order and total-order Sobol indices.

        Uses Saltelli's sampling scheme for efficiency.
        """
        A, B = self.generate_samples(bounds_factor)
        n = self.model.n_barriers

        # Evaluate model at A and B
        Y_A = np.array([self.model.calculate_success(a) for a in A])
        Y_B = np.array([self.model.calculate_success(b) for b in B])

        # Total variance
        Y_all = np.concatenate([Y_A, Y_B])
        V_Y = np.var(Y_all)

        if V_Y < 1e-15:
            # No variance - model is constant
            return pd.DataFrame({
                'barrier_idx': range(n),
                'S1': [0] * n,
                'ST': [0] * n,
                'S1_conf': [0] * n,
                'ST_conf': [0] * n
            })

        first_order = []
        total_order = []

        for i in range(n):
            # Create AB_i matrix (A with i-th column from B)
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            Y_AB_i = np.array([self.model.calculate_success(ab) for ab in AB_i])

            # First-order index: S_i = V[E(Y|X_i)] / V(Y)
            # Estimated using: S_i ≈ (1/N) * Σ Y_B * (Y_AB_i - Y_A) / V(Y)
            S1_i = np.mean(Y_B * (Y_AB_i - Y_A)) / V_Y

            # Total-order index: S_Ti = E[V(Y|X_~i)] / V(Y)
            # Estimated using: S_Ti ≈ (1/2N) * Σ (Y_A - Y_AB_i)^2 / V(Y)
            ST_i = 0.5 * np.mean((Y_A - Y_AB_i)**2) / V_Y

            first_order.append(max(0, S1_i))  # Clip negative values
            total_order.append(max(0, ST_i))

        # Bootstrap confidence intervals
        n_bootstrap = 100
        S1_boot = np.zeros((n_bootstrap, n))
        ST_boot = np.zeros((n_bootstrap, n))

        for b in range(n_bootstrap):
            idx = np.random.choice(self.n_samples, self.n_samples, replace=True)
            Y_A_b = Y_A[idx]
            Y_B_b = Y_B[idx]
            V_Y_b = np.var(np.concatenate([Y_A_b, Y_B_b]))

            if V_Y_b > 1e-15:
                for i in range(n):
                    AB_i = A[idx].copy()
                    AB_i[:, i] = B[idx, i]
                    Y_AB_i_b = np.array([self.model.calculate_success(ab) for ab in AB_i])

                    S1_boot[b, i] = np.mean(Y_B_b * (Y_AB_i_b - Y_A_b)) / V_Y_b
                    ST_boot[b, i] = 0.5 * np.mean((Y_A_b - Y_AB_i_b)**2) / V_Y_b

        S1_conf = np.std(S1_boot, axis=0) * 1.96
        ST_conf = np.std(ST_boot, axis=0) * 1.96

        return pd.DataFrame({
            'barrier_idx': range(n),
            'S1': first_order,
            'ST': total_order,
            'S1_conf': S1_conf,
            'ST_conf': ST_conf,
            'interaction': np.array(total_order) - np.array(first_order)
        })


# =============================================================================
# MORRIS ELEMENTARY EFFECTS METHOD
# =============================================================================

class MorrisScreening:
    """
    Morris method for elementary effects screening.

    Efficient global sensitivity method that requires fewer model evaluations
    than Sobol. Identifies which parameters have:
    - Negligible effects
    - Linear effects
    - Nonlinear or interaction effects
    """

    def __init__(self, model: BarrierModel, n_trajectories: int = 50):
        self.model = model
        self.n_trajectories = n_trajectories
        self.baseline = model.barrier_probs.copy()

    def generate_trajectory(self, levels: int = 4, bounds_factor: float = 0.3) -> np.ndarray:
        """Generate a single Morris trajectory through parameter space."""
        n = self.model.n_barriers

        # Define grid
        lower = np.maximum(0.05, self.baseline * (1 - bounds_factor))
        upper = np.minimum(0.95, self.baseline * (1 + bounds_factor))

        # Random starting point
        x = np.random.randint(0, levels, n) / (levels - 1)
        x = lower + x * (upper - lower)

        # Generate trajectory by perturbing one dimension at a time
        trajectory = [x.copy()]
        delta = (upper - lower) / (levels - 1)

        order = np.random.permutation(n)
        for i in order:
            x_new = x.copy()
            if np.random.random() < 0.5:
                x_new[i] = min(upper[i], x[i] + delta[i])
            else:
                x_new[i] = max(lower[i], x[i] - delta[i])
            trajectory.append(x_new.copy())
            x = x_new

        return np.array(trajectory)

    def calculate_elementary_effects(self, bounds_factor: float = 0.3) -> pd.DataFrame:
        """
        Calculate elementary effects for all parameters.

        Returns:
            mu: Mean of elementary effects (overall importance)
            mu_star: Mean of absolute elementary effects (importance without sign)
            sigma: Std of elementary effects (nonlinearity/interaction indicator)
        """
        n = self.model.n_barriers
        effects = {i: [] for i in range(n)}

        lower = np.maximum(0.05, self.baseline * (1 - bounds_factor))
        upper = np.minimum(0.95, self.baseline * (1 + bounds_factor))
        delta = (upper - lower) / 3  # 4 levels

        for _ in range(self.n_trajectories):
            trajectory = self.generate_trajectory(bounds_factor=bounds_factor)

            # Evaluate at each point
            outputs = [self.model.calculate_success(t) for t in trajectory]

            # Calculate elementary effects
            for j in range(1, len(trajectory)):
                # Find which parameter changed
                diff = trajectory[j] - trajectory[j-1]
                changed_idx = np.argmax(np.abs(diff))

                if np.abs(diff[changed_idx]) > 1e-10:
                    # Elementary effect
                    EE = (outputs[j] - outputs[j-1]) / diff[changed_idx]
                    effects[changed_idx].append(EE)

        # Calculate statistics
        results = []
        for i in range(n):
            if len(effects[i]) > 0:
                ee_array = np.array(effects[i])
                results.append({
                    'barrier_idx': i,
                    'mu': np.mean(ee_array),
                    'mu_star': np.mean(np.abs(ee_array)),
                    'sigma': np.std(ee_array),
                    'n_effects': len(ee_array)
                })
            else:
                results.append({
                    'barrier_idx': i,
                    'mu': 0,
                    'mu_star': 0,
                    'sigma': 0,
                    'n_effects': 0
                })

        return pd.DataFrame(results)


# =============================================================================
# SIGNAL-TO-NOISE RATIO ANALYSIS
# =============================================================================

class SNRAnalysis:
    """
    Signal-to-Noise Ratio analysis for model robustness.

    Evaluates how robust model conclusions are to:
    - Parameter uncertainty
    - Measurement noise
    - Stochastic variation
    """

    def __init__(self, model: BarrierModel, n_simulations: int = 1000):
        self.model = model
        self.n_simulations = n_simulations
        self.baseline = model.barrier_probs.copy()

    def noise_injection_analysis(self,
                                  noise_levels: np.ndarray = None) -> pd.DataFrame:
        """
        Analyze model output stability under parameter noise.

        Injects Gaussian noise at various levels and measures output variance.
        """
        if noise_levels is None:
            noise_levels = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])

        results = []
        baseline_output = self.model.calculate_success()

        for noise_std in noise_levels:
            outputs = []

            for _ in range(self.n_simulations):
                # Add multiplicative Gaussian noise
                noisy_probs = self.baseline * (1 + np.random.normal(0, noise_std, self.model.n_barriers))
                noisy_probs = np.clip(noisy_probs, 0.01, 0.99)

                output = self.model.calculate_success(noisy_probs)
                outputs.append(output)

            outputs = np.array(outputs)

            # Signal = mean output, Noise = std of output
            signal = np.mean(outputs)
            noise = np.std(outputs)
            snr = signal / noise if noise > 0 else np.inf
            snr_db = 10 * np.log10(snr) if snr > 0 and snr < np.inf else np.nan

            results.append({
                'noise_level': noise_std,
                'mean_output': signal,
                'std_output': noise,
                'cv': noise / signal if signal > 0 else np.inf,
                'snr': snr,
                'snr_db': snr_db,
                'ci_lower': np.percentile(outputs, 2.5),
                'ci_upper': np.percentile(outputs, 97.5),
                'output_range': np.ptp(outputs)
            })

        return pd.DataFrame(results)

    def key_finding_robustness(self, n_bootstrap: int = 1000) -> Dict:
        """
        Test robustness of key findings to parameter uncertainty.

        Key findings tested:
        1. Three-way interaction dominance (>80%)
        2. Individual barrier effects near zero
        3. Complete removal required for >90% success
        """
        results = {
            'three_way_interaction': [],
            'max_individual_effect': [],
            'barriers_for_90pct': []
        }

        for _ in range(n_bootstrap):
            # Sample parameters with 10% uncertainty
            noisy_probs = self.baseline * (1 + np.random.normal(0, 0.10, self.model.n_barriers))
            noisy_probs = np.clip(noisy_probs, 0.01, 0.99)

            noisy_model = BarrierModel(noisy_probs)
            baseline = noisy_model.calculate_success()

            # 1. Calculate individual effects
            individual_effects = []
            for i in range(self.model.n_barriers):
                effect = noisy_model.calculate_success_removing([i]) - baseline
                individual_effects.append(effect)

            results['max_individual_effect'].append(max(individual_effects))

            # 2. Calculate layer effects
            layer_effects = {}
            for layer, indices in noisy_model.layer_indices.items():
                effect = noisy_model.calculate_success_removing(indices) - baseline
                layer_effects[layer] = effect

            # Pairwise and three-way
            all_indices = list(range(self.model.n_barriers))
            full_effect = noisy_model.calculate_success_removing(all_indices) - baseline

            sum_individual_layers = sum(layer_effects.values())

            # Three-way interaction (simplified calculation)
            three_way = full_effect - sum_individual_layers
            three_way_pct = (three_way / full_effect * 100) if full_effect > 0 else 0
            results['three_way_interaction'].append(three_way_pct)

            # 3. How many barriers needed for 90%?
            for n_removed in range(1, 12):
                # Try removing n barriers (greedy by baseline probability)
                sorted_indices = np.argsort(noisy_probs)[:n_removed]
                success = noisy_model.calculate_success_removing(list(sorted_indices))
                if success >= 0.90:
                    results['barriers_for_90pct'].append(n_removed)
                    break
            else:
                results['barriers_for_90pct'].append(11)

        # Summarize
        summary = {
            'three_way_interaction': {
                'mean': np.mean(results['three_way_interaction']),
                'std': np.std(results['three_way_interaction']),
                'ci_lower': np.percentile(results['three_way_interaction'], 2.5),
                'ci_upper': np.percentile(results['three_way_interaction'], 97.5),
                'finding_robust': np.mean(np.array(results['three_way_interaction']) > 70)
            },
            'max_individual_effect': {
                'mean': np.mean(results['max_individual_effect']),
                'std': np.std(results['max_individual_effect']),
                'ci_lower': np.percentile(results['max_individual_effect'], 2.5),
                'ci_upper': np.percentile(results['max_individual_effect'], 97.5),
                'finding_robust': np.mean(np.array(results['max_individual_effect']) < 0.01)
            },
            'barriers_for_90pct': {
                'mean': np.mean(results['barriers_for_90pct']),
                'std': np.std(results['barriers_for_90pct']),
                'mode': stats.mode(results['barriers_for_90pct'], keepdims=False)[0],
                'finding_robust': np.mean(np.array(results['barriers_for_90pct']) >= 10)
            }
        }

        return summary


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

class BootstrapValidation:
    """
    Bootstrap-based validation and confidence interval estimation.
    """

    def __init__(self, model: BarrierModel, n_bootstrap: int = 2000):
        self.model = model
        self.n_bootstrap = n_bootstrap
        self.baseline = model.barrier_probs.copy()

    def bootstrap_success_probability(self) -> Dict:
        """Bootstrap CI for baseline success probability."""
        outputs = []

        for _ in range(self.n_bootstrap):
            # Resample with uncertainty
            boot_probs = self.baseline * (1 + np.random.normal(0, 0.05, self.model.n_barriers))
            boot_probs = np.clip(boot_probs, 0.01, 0.99)
            outputs.append(BarrierModel(boot_probs).calculate_success())

        outputs = np.array(outputs)

        return {
            'point_estimate': self.model.calculate_success(),
            'bootstrap_mean': np.mean(outputs),
            'bootstrap_std': np.std(outputs),
            'ci_95_lower': np.percentile(outputs, 2.5),
            'ci_95_upper': np.percentile(outputs, 97.5),
            'ci_99_lower': np.percentile(outputs, 0.5),
            'ci_99_upper': np.percentile(outputs, 99.5)
        }

    def bootstrap_barrier_removal_effects(self) -> pd.DataFrame:
        """Bootstrap CIs for all barrier removal effects."""
        results = []

        for i in range(self.model.n_barriers):
            effects = []

            for _ in range(self.n_bootstrap):
                boot_probs = self.baseline * (1 + np.random.normal(0, 0.05, self.model.n_barriers))
                boot_probs = np.clip(boot_probs, 0.01, 0.99)

                boot_model = BarrierModel(boot_probs)
                baseline = boot_model.calculate_success()
                removed = boot_model.calculate_success_removing([i])
                effects.append(removed - baseline)

            effects = np.array(effects)

            results.append({
                'barrier_idx': i,
                'effect_mean': np.mean(effects),
                'effect_std': np.std(effects),
                'ci_lower': np.percentile(effects, 2.5),
                'ci_upper': np.percentile(effects, 97.5),
                'significant': not (np.percentile(effects, 2.5) <= 0 <= np.percentile(effects, 97.5))
            })

        return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_sensitivity_analysis(oat_results: pd.DataFrame,
                              sobol_results: pd.DataFrame,
                              morris_results: pd.DataFrame,
                              save_path: str = None):
    """Create comprehensive sensitivity analysis figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    barrier_names = [
        'Rapid Trans', 'Multi-Sys', 'Perm Store',
        'Error Det', 'Correct Proc', 'Incomp Prop',
        'Aware Gap', 'Record Acc', 'Legal Know', 'Legal Res', 'Sys Bias'
    ]

    # Panel A: OAT Spider Plot
    ax = axes[0, 0]
    oat_sensitivity = oat_results.groupby('barrier_idx').apply(
        lambda x: np.std(x['output_change_pct'])
    ).values

    angles = np.linspace(0, 2 * np.pi, len(barrier_names), endpoint=False).tolist()
    angles += angles[:1]
    oat_sensitivity_plot = list(oat_sensitivity) + [oat_sensitivity[0]]

    ax = plt.subplot(2, 2, 1, projection='polar')
    ax.plot(angles, oat_sensitivity_plot, 'o-', linewidth=2, color='blue')
    ax.fill(angles, oat_sensitivity_plot, alpha=0.25, color='blue')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(barrier_names, fontsize=8)
    ax.set_title('A. OAT Sensitivity (Std of Output Change)', fontsize=12, fontweight='bold', pad=20)

    # Panel B: Sobol Indices
    ax = axes[0, 1]
    x = np.arange(len(barrier_names))
    width = 0.35

    ax.bar(x - width/2, sobol_results['S1'], width, label='First-order (S1)', color='blue', alpha=0.7)
    ax.bar(x + width/2, sobol_results['ST'], width, label='Total-order (ST)', color='red', alpha=0.7)
    ax.errorbar(x - width/2, sobol_results['S1'], yerr=sobol_results['S1_conf'], fmt='none', color='black', capsize=2)
    ax.errorbar(x + width/2, sobol_results['ST'], yerr=sobol_results['ST_conf'], fmt='none', color='black', capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels(barrier_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Sobol Index')
    ax.set_title('B. Global Sensitivity (Sobol Indices)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: Morris Screening
    ax = axes[1, 0]
    ax.scatter(morris_results['mu_star'], morris_results['sigma'],
               s=100, c=range(len(barrier_names)), cmap='tab10', alpha=0.7)

    for i, name in enumerate(barrier_names):
        ax.annotate(name, (morris_results['mu_star'].iloc[i], morris_results['sigma'].iloc[i]),
                   fontsize=8, xytext=(5, 5), textcoords='offset points')

    ax.axhline(y=np.mean(morris_results['sigma']), color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=np.mean(morris_results['mu_star']), color='red', linestyle='--', alpha=0.5)

    ax.set_xlabel('μ* (Mean Absolute Effect)', fontsize=10)
    ax.set_ylabel('σ (Effect Std Dev)', fontsize=10)
    ax.set_title('C. Morris Screening (μ* vs σ)\nHigh σ = Nonlinear/Interaction Effects',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel D: Interaction Contribution
    ax = axes[1, 1]
    interaction = sobol_results['ST'] - sobol_results['S1']
    colors = ['green' if i > 0.01 else 'gray' for i in interaction]

    ax.barh(range(len(barrier_names)), interaction, color=colors, alpha=0.7)
    ax.set_yticks(range(len(barrier_names)))
    ax.set_yticklabels(barrier_names, fontsize=8)
    ax.set_xlabel('Interaction Contribution (ST - S1)', fontsize=10)
    ax.set_title('D. Parameter Interaction Effects', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_snr_analysis(snr_results: pd.DataFrame,
                      robustness_results: Dict,
                      save_path: str = None):
    """Create SNR and robustness analysis figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: SNR vs Noise Level
    ax = axes[0, 0]
    ax.plot(snr_results['noise_level'] * 100, snr_results['snr_db'], 'o-',
            linewidth=2, markersize=8, color='blue')
    ax.axhline(y=10, color='green', linestyle='--', label='10 dB (Good)')
    ax.axhline(y=3, color='orange', linestyle='--', label='3 dB (Marginal)')
    ax.axhline(y=0, color='red', linestyle='--', label='0 dB (Poor)')

    ax.set_xlabel('Parameter Noise Level (%)', fontsize=12)
    ax.set_ylabel('Signal-to-Noise Ratio (dB)', fontsize=12)
    ax.set_title('A. Model Robustness: SNR vs Parameter Uncertainty', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Output Distribution at Different Noise Levels
    ax = axes[0, 1]
    ax.fill_between(snr_results['noise_level'] * 100,
                    snr_results['ci_lower'] * 100,
                    snr_results['ci_upper'] * 100,
                    alpha=0.3, color='blue', label='95% CI')
    ax.plot(snr_results['noise_level'] * 100, snr_results['mean_output'] * 100,
            'o-', linewidth=2, color='blue', label='Mean')

    ax.set_xlabel('Parameter Noise Level (%)', fontsize=12)
    ax.set_ylabel('Success Probability (%)', fontsize=12)
    ax.set_title('B. Output Uncertainty Propagation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel C: Key Finding Robustness
    ax = axes[1, 0]
    findings = ['Three-way\nInteraction\n>70%', 'Max Individual\nEffect\n<1%', 'Need 10+\nBarriers\nfor 90%']
    robustness_scores = [
        robustness_results['three_way_interaction']['finding_robust'] * 100,
        robustness_results['max_individual_effect']['finding_robust'] * 100,
        robustness_results['barriers_for_90pct']['finding_robust'] * 100
    ]

    colors = ['green' if r >= 95 else 'orange' if r >= 80 else 'red' for r in robustness_scores]
    bars = ax.bar(findings, robustness_scores, color=colors, alpha=0.7, edgecolor='black')

    ax.axhline(y=95, color='green', linestyle='--', label='95% threshold')
    ax.axhline(y=80, color='orange', linestyle='--', label='80% threshold')

    for bar, score in zip(bars, robustness_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{score:.1f}%', ha='center', fontsize=10, fontweight='bold')

    ax.set_ylabel('% of Bootstrap Samples Supporting Finding', fontsize=10)
    ax.set_title('C. Key Finding Robustness Under 10% Parameter Uncertainty',
                fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel D: Coefficient of Variation
    ax = axes[1, 1]
    ax.plot(snr_results['noise_level'] * 100, snr_results['cv'] * 100, 'o-',
            linewidth=2, markersize=8, color='purple')
    ax.axhline(y=10, color='green', linestyle='--', label='CV < 10% (Excellent)')
    ax.axhline(y=25, color='orange', linestyle='--', label='CV < 25% (Acceptable)')
    ax.axhline(y=50, color='red', linestyle='--', label='CV > 50% (Poor)')

    ax.set_xlabel('Parameter Noise Level (%)', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=12)
    ax.set_title('D. Output Variability (CV) vs Input Uncertainty', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run comprehensive sensitivity and robustness analysis."""
    print("=" * 80)
    print("SENSITIVITY ANALYSIS AND MODEL VALIDATION")
    print("Algorithmic Bias Epidemiology Framework")
    print("=" * 80)

    # Initialize model
    model = BarrierModel()
    print(f"\nBaseline success probability: {model.calculate_success()*100:.4f}%")

    # ==========================================================================
    # 1. OAT Sensitivity Analysis
    # ==========================================================================
    print("\n" + "-" * 40)
    print("1. ONE-AT-A-TIME (OAT) SENSITIVITY")
    print("-" * 40)

    oat = OATSensitivityAnalysis(model)
    oat_results = oat.analyze()
    oat_indices = oat.calculate_sensitivity_indices()

    print("\nNormalized Sensitivity Indices (|S_i|):")
    for _, row in oat_indices.head(5).iterrows():
        print(f"  Barrier {int(row['barrier_idx'])}: {row['abs_sensitivity']:.4f}")

    # ==========================================================================
    # 2. Sobol Global Sensitivity
    # ==========================================================================
    print("\n" + "-" * 40)
    print("2. SOBOL GLOBAL SENSITIVITY ANALYSIS")
    print("-" * 40)

    sobol = SobolSensitivityAnalysis(model, n_samples=5000)
    sobol_results = sobol.calculate_indices()

    print("\nSobol Indices (with 95% CI):")
    print(f"{'Barrier':<10} {'S1':>8} {'S1_CI':>10} {'ST':>8} {'ST_CI':>10}")
    for _, row in sobol_results.iterrows():
        print(f"  {int(row['barrier_idx']):<8} {row['S1']:>8.4f} ±{row['S1_conf']:>8.4f} "
              f"{row['ST']:>8.4f} ±{row['ST_conf']:>8.4f}")

    # ==========================================================================
    # 3. Morris Screening
    # ==========================================================================
    print("\n" + "-" * 40)
    print("3. MORRIS ELEMENTARY EFFECTS SCREENING")
    print("-" * 40)

    morris = MorrisScreening(model, n_trajectories=100)
    morris_results = morris.calculate_elementary_effects()

    print("\nMorris Statistics:")
    print(f"{'Barrier':<10} {'μ':>10} {'μ*':>10} {'σ':>10}")
    for _, row in morris_results.iterrows():
        print(f"  {int(row['barrier_idx']):<8} {row['mu']:>10.6f} {row['mu_star']:>10.6f} {row['sigma']:>10.6f}")

    # ==========================================================================
    # 4. SNR Analysis
    # ==========================================================================
    print("\n" + "-" * 40)
    print("4. SIGNAL-TO-NOISE RATIO ANALYSIS")
    print("-" * 40)

    snr = SNRAnalysis(model, n_simulations=2000)
    snr_results = snr.noise_injection_analysis()

    print("\nSNR at Different Noise Levels:")
    print(f"{'Noise %':<10} {'Mean':>10} {'Std':>10} {'SNR (dB)':>10} {'CV %':>10}")
    for _, row in snr_results.iterrows():
        print(f"  {row['noise_level']*100:<8.0f} {row['mean_output']*100:>10.4f} "
              f"{row['std_output']*100:>10.4f} {row['snr_db']:>10.2f} {row['cv']*100:>10.2f}")

    # ==========================================================================
    # 5. Key Finding Robustness
    # ==========================================================================
    print("\n" + "-" * 40)
    print("5. KEY FINDING ROBUSTNESS")
    print("-" * 40)

    robustness = snr.key_finding_robustness(n_bootstrap=1000)

    print("\nThree-way Interaction Dominance:")
    r = robustness['three_way_interaction']
    print(f"  Mean: {r['mean']:.1f}% (95% CI: {r['ci_lower']:.1f}% - {r['ci_upper']:.1f}%)")
    print(f"  Finding robust (>70%): {r['finding_robust']*100:.1f}% of bootstrap samples")

    print("\nIndividual Barrier Effects Near Zero:")
    r = robustness['max_individual_effect']
    print(f"  Max individual effect mean: {r['mean']*100:.4f}%")
    print(f"  Finding robust (<1%): {r['finding_robust']*100:.1f}% of bootstrap samples")

    print("\nBarriers Needed for 90% Success:")
    r = robustness['barriers_for_90pct']
    print(f"  Mean: {r['mean']:.1f} barriers, Mode: {r['mode']} barriers")
    print(f"  Finding robust (≥10): {r['finding_robust']*100:.1f}% of bootstrap samples")

    # ==========================================================================
    # 6. Bootstrap Confidence Intervals
    # ==========================================================================
    print("\n" + "-" * 40)
    print("6. BOOTSTRAP CONFIDENCE INTERVALS")
    print("-" * 40)

    bootstrap = BootstrapValidation(model, n_bootstrap=2000)
    boot_success = bootstrap.bootstrap_success_probability()

    print("\nBaseline Success Probability:")
    print(f"  Point estimate: {boot_success['point_estimate']*100:.4f}%")
    print(f"  Bootstrap mean: {boot_success['bootstrap_mean']*100:.4f}%")
    print(f"  95% CI: ({boot_success['ci_95_lower']*100:.4f}%, {boot_success['ci_95_upper']*100:.4f}%)")

    # ==========================================================================
    # Generate Visualizations
    # ==========================================================================
    print("\n" + "-" * 40)
    print("GENERATING VISUALIZATIONS")
    print("-" * 40)

    plot_sensitivity_analysis(oat_results, sobol_results, morris_results,
                             'sensitivity_analysis.png')
    plot_snr_analysis(snr_results, robustness, 'snr_robustness.png')

    # ==========================================================================
    # Export Results
    # ==========================================================================
    print("\n" + "-" * 40)
    print("EXPORTING RESULTS")
    print("-" * 40)

    oat_results.to_csv('oat_sensitivity.csv', index=False)
    oat_indices.to_csv('oat_indices.csv', index=False)
    sobol_results.to_csv('sobol_indices.csv', index=False)
    morris_results.to_csv('morris_screening.csv', index=False)
    snr_results.to_csv('snr_analysis.csv', index=False)

    # Robustness summary
    robustness_df = pd.DataFrame([
        {'finding': 'three_way_interaction', **robustness['three_way_interaction']},
        {'finding': 'max_individual_effect', **robustness['max_individual_effect']},
        {'finding': 'barriers_for_90pct', **robustness['barriers_for_90pct']}
    ])
    robustness_df.to_csv('robustness_summary.csv', index=False)

    print("\nFiles exported:")
    print("  - sensitivity_analysis.png")
    print("  - snr_robustness.png")
    print("  - oat_sensitivity.csv, oat_indices.csv")
    print("  - sobol_indices.csv")
    print("  - morris_screening.csv")
    print("  - snr_analysis.csv")
    print("  - robustness_summary.csv")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
