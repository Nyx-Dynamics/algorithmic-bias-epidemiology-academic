#!/usr/bin/env python3
"""
COUNTERFACTUAL BARRIER ANALYSIS FOR ALGORITHMIC BIAS
=====================================================

Analyzes the effect of removing barriers through counterfactual modeling:

1. INDIVIDUAL REMOVAL: Remove one barrier at a time
   - Measures marginal effect of each barrier
   - Identifies highest-impact barriers

2. STEPWISE CUMULATIVE REMOVAL: Remove barriers progressively
   - Forward: Remove barriers in order (first → last)
   - Backward: Remove barriers in reverse (last → first)
   - Optimal: Remove in order of impact

3. BARRIER SET/GROUP REMOVAL: Remove entire layers
   - Layer 1: Data Integration Barriers
   - Layer 2: Data Accuracy Barriers
   - Layer 3: Institutional Barriers
   - Combinations of layers

EPIDEMIOLOGICAL PARALLEL:
    - Population Attributable Fraction (PAF) for each barrier
    - Intervention impact analysis
    - Cascade interruption effects

MATHEMATICAL FRAMEWORK:
    Let P(Success | Barriers) = baseline probability
    Let P(Success | Barriers - {b}) = probability without barrier b

    Individual Effect(b) = P(Success | Barriers - {b}) - P(Success | Barriers)
    Cumulative Effect({b1,...,bn}) = P(Success | Barriers - {b1,...,bn}) - P(Success | Barriers)

    Population Attributable Fraction (PAF):
    PAF(b) = [P(Success | no b) - P(Success)] / [1 - P(Success)]

Author: Nyx Dynamics LLC / AC Demidont, DO
Development Tool: Claude Code (claude-opus-4-5-20251101)
Date: January 8, 2026
"""

import numpy as np
from scipy.special import expit
from typing import Dict, List, Tuple, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations, permutations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict


# =============================================================================
# PART I: BARRIER DEFINITIONS
# =============================================================================

class BarrierLayer(Enum):
    """Three layers of barriers (parallel to HIV architectural barriers)."""
    DATA_INTEGRATION = 1    # Barriers to data entering the system
    DATA_ACCURACY = 2       # Barriers to data being correct
    INSTITUTIONAL = 3       # Barriers to challenging/correcting


class BarrierType(Enum):
    """Types of barriers within each layer."""
    # Layer 1: Data Integration
    LACK_OF_AWARENESS = auto()
    NO_DATA_ACCESS = auto()
    TRANSMISSION_SPEED = auto()

    # Layer 2: Data Accuracy
    ERROR_DETECTION_DIFFICULTY = auto()
    CORRECTION_COMPLEXITY = auto()
    VERIFICATION_BURDEN = auto()

    # Layer 3: Institutional
    LEGAL_KNOWLEDGE_GAP = auto()
    COST_OF_CHALLENGE = auto()
    TIME_TO_RESOLUTION = auto()
    RETALIATION_RISK = auto()
    SYSTEMIC_BIAS = auto()


@dataclass
class Barrier:
    """A single barrier in the algorithmic bias cascade."""
    name: str
    barrier_type: BarrierType
    layer: BarrierLayer
    base_probability: float  # P(barrier blocks success)
    description: str
    removable: bool = True
    removal_cost_usd: float = 0.0
    removal_time_months: float = 0.0

    def get_blocking_probability(self, context: Dict = None) -> float:
        """Get probability this barrier blocks success given context."""
        p = self.base_probability
        if context:
            # Adjust based on context factors
            if context.get("has_legal_counsel"):
                if self.layer == BarrierLayer.INSTITUTIONAL:
                    p *= 0.5
            if context.get("high_income"):
                if "COST" in self.barrier_type.name:
                    p *= 0.3
            if context.get("tech_savvy"):
                if self.layer == BarrierLayer.DATA_INTEGRATION:
                    p *= 0.6
        return min(0.99, max(0.01, p))


@dataclass
class BarrierSet:
    """A complete set of barriers for analysis."""
    barriers: List[Barrier]
    name: str = "Default Barrier Set"

    def get_barriers_by_layer(self, layer: BarrierLayer) -> List[Barrier]:
        return [b for b in self.barriers if b.layer == layer]

    def get_barrier_by_type(self, barrier_type: BarrierType) -> Optional[Barrier]:
        for b in self.barriers:
            if b.barrier_type == barrier_type:
                return b
        return None


# =============================================================================
# PART II: DEFAULT BARRIER CONFIGURATION
# =============================================================================

def create_default_barrier_set() -> BarrierSet:
    """
    Create the default barrier set based on the 3-layer model.

    Probabilities derived from empirical estimates and the
    AIDS and Behavior architectural barriers framework.
    """
    barriers = [
        # Layer 1: Data Integration Barriers
        Barrier(
            name="Awareness Gap",
            barrier_type=BarrierType.LACK_OF_AWARENESS,
            layer=BarrierLayer.DATA_INTEGRATION,
            base_probability=0.70,
            description="Individual unaware that adverse data was recorded",
            removal_cost_usd=0,
            removal_time_months=0.5
        ),
        Barrier(
            name="Data Access Denial",
            barrier_type=BarrierType.NO_DATA_ACCESS,
            layer=BarrierLayer.DATA_INTEGRATION,
            base_probability=0.45,
            description="Cannot access own records to verify accuracy",
            removal_cost_usd=50,
            removal_time_months=1.0
        ),
        Barrier(
            name="Rapid Transmission",
            barrier_type=BarrierType.TRANSMISSION_SPEED,
            layer=BarrierLayer.DATA_INTEGRATION,
            base_probability=0.80,
            description="Data propagates faster than correction can occur",
            removal_cost_usd=500,
            removal_time_months=0.25
        ),

        # Layer 2: Data Accuracy Barriers
        Barrier(
            name="Error Detection Difficulty",
            barrier_type=BarrierType.ERROR_DETECTION_DIFFICULTY,
            layer=BarrierLayer.DATA_ACCURACY,
            base_probability=0.55,
            description="Difficult to identify which data is erroneous",
            removal_cost_usd=200,
            removal_time_months=2.0
        ),
        Barrier(
            name="Correction Process Complexity",
            barrier_type=BarrierType.CORRECTION_COMPLEXITY,
            layer=BarrierLayer.DATA_ACCURACY,
            base_probability=0.65,
            description="Complex multi-step process to correct data",
            removal_cost_usd=300,
            removal_time_months=3.0
        ),
        Barrier(
            name="Verification Burden",
            barrier_type=BarrierType.VERIFICATION_BURDEN,
            layer=BarrierLayer.DATA_ACCURACY,
            base_probability=0.50,
            description="Burden of proof falls on individual, not data holder",
            removal_cost_usd=400,
            removal_time_months=2.0
        ),

        # Layer 3: Institutional Barriers
        Barrier(
            name="Legal Knowledge Gap",
            barrier_type=BarrierType.LEGAL_KNOWLEDGE_GAP,
            layer=BarrierLayer.INSTITUTIONAL,
            base_probability=0.75,
            description="Unaware of legal rights and remedies (FCRA, etc.)",
            removal_cost_usd=0,
            removal_time_months=1.0
        ),
        Barrier(
            name="Cost of Legal Challenge",
            barrier_type=BarrierType.COST_OF_CHALLENGE,
            layer=BarrierLayer.INSTITUTIONAL,
            base_probability=0.60,
            description="Cannot afford legal representation or filing fees",
            removal_cost_usd=2000,
            removal_time_months=0.5
        ),
        Barrier(
            name="Time to Resolution",
            barrier_type=BarrierType.TIME_TO_RESOLUTION,
            layer=BarrierLayer.INSTITUTIONAL,
            base_probability=0.70,
            description="Resolution takes too long, damage already done",
            removal_cost_usd=0,
            removal_time_months=6.0
        ),
        Barrier(
            name="Retaliation Risk",
            barrier_type=BarrierType.RETALIATION_RISK,
            layer=BarrierLayer.INSTITUTIONAL,
            base_probability=0.40,
            description="Fear of retaliation prevents challenge",
            removal_cost_usd=1000,
            removal_time_months=0
        ),
        Barrier(
            name="Systemic Bias",
            barrier_type=BarrierType.SYSTEMIC_BIAS,
            layer=BarrierLayer.INSTITUTIONAL,
            base_probability=0.85,
            description="System designed to favor data holders over individuals",
            removal_cost_usd=10000,
            removal_time_months=24.0
        ),
    ]

    return BarrierSet(barriers=barriers, name="Algorithmic Bias Barrier Set")


# =============================================================================
# PART III: COUNTERFACTUAL MODEL
# =============================================================================

class CounterfactualBarrierModel:
    """
    Model for counterfactual analysis of barrier removal.

    Calculates success probability under different barrier configurations
    using a multiplicative model (each barrier independently blocks success).
    """

    def __init__(self, barrier_set: BarrierSet = None):
        self.barrier_set = barrier_set or create_default_barrier_set()
        self.baseline_success_rate = None
        self._cache = {}

    def calculate_success_probability(self,
                                       active_barriers: List[Barrier],
                                       context: Dict = None,
                                       model: str = "multiplicative") -> float:
        """
        Calculate P(Success) given active barriers.

        Models:
        - "multiplicative": P(Success) = ∏(1 - P(barrier blocks))
        - "additive": P(Success) = 1 - min(1, Σ P(barrier blocks))
        - "max": P(Success) = 1 - max(P(barrier blocks))
        """
        if not active_barriers:
            return 0.95  # Near-certain success with no barriers

        blocking_probs = [b.get_blocking_probability(context) for b in active_barriers]

        if model == "multiplicative":
            # Each barrier independently can block success
            p_success = 1.0
            for p_block in blocking_probs:
                p_success *= (1 - p_block)
            return p_success

        elif model == "additive":
            # Barriers add up (capped at 1)
            total_block = min(1.0, sum(blocking_probs) / len(blocking_probs))
            return 1 - total_block

        elif model == "max":
            # Worst barrier determines outcome
            return 1 - max(blocking_probs)

        else:
            raise ValueError(f"Unknown model: {model}")

    def get_baseline(self, context: Dict = None, model: str = "multiplicative") -> float:
        """Get baseline success probability with all barriers."""
        self.baseline_success_rate = self.calculate_success_probability(
            self.barrier_set.barriers, context, model
        )
        return self.baseline_success_rate

    def remove_barrier(self,
                       barrier: Barrier,
                       context: Dict = None,
                       model: str = "multiplicative") -> Dict:
        """
        Calculate effect of removing a single barrier.

        Returns dict with:
        - new_success_rate: P(Success) without this barrier
        - absolute_change: Change in success probability
        - relative_change: Percentage increase in success
        - paf: Population Attributable Fraction
        """
        baseline = self.get_baseline(context, model)

        remaining = [b for b in self.barrier_set.barriers if b != barrier]
        new_rate = self.calculate_success_probability(remaining, context, model)

        absolute_change = new_rate - baseline
        relative_change = (new_rate - baseline) / baseline if baseline > 0 else float('inf')

        # PAF: What fraction of failures are attributable to this barrier?
        # PAF = (P_without - P_with) / (1 - P_with)
        paf = absolute_change / (1 - baseline) if baseline < 1 else 0

        return {
            "barrier": barrier.name,
            "barrier_type": barrier.barrier_type.name,
            "layer": barrier.layer.name,
            "baseline_success": baseline,
            "new_success_rate": new_rate,
            "absolute_change": absolute_change,
            "relative_change": relative_change,
            "paf": paf,
            "removal_cost": barrier.removal_cost_usd,
            "removal_time": barrier.removal_time_months,
            "cost_effectiveness": absolute_change / max(1, barrier.removal_cost_usd) * 1000
        }


# =============================================================================
# PART IV: INDIVIDUAL BARRIER REMOVAL ANALYSIS
# =============================================================================

class IndividualBarrierAnalysis:
    """
    Analyze effect of removing each barrier individually.

    Like analyzing single-gene knockouts in biology, or
    single-intervention removal in epidemiology.
    """

    def __init__(self, model: CounterfactualBarrierModel):
        self.model = model

    def analyze_all_barriers(self,
                             context: Dict = None,
                             calc_model: str = "multiplicative") -> List[Dict]:
        """Analyze effect of removing each barrier one at a time."""
        results = []

        for barrier in self.model.barrier_set.barriers:
            result = self.model.remove_barrier(barrier, context, calc_model)
            results.append(result)

        # Sort by absolute change (highest impact first)
        results.sort(key=lambda x: x["absolute_change"], reverse=True)

        return results

    def get_top_barriers(self, n: int = 5, context: Dict = None) -> List[Dict]:
        """Get the n barriers with highest individual impact."""
        all_results = self.analyze_all_barriers(context)
        return all_results[:n]

    def get_barriers_by_layer(self, context: Dict = None) -> Dict[str, List[Dict]]:
        """Group barrier impacts by layer."""
        all_results = self.analyze_all_barriers(context)

        by_layer = {
            "DATA_INTEGRATION": [],
            "DATA_ACCURACY": [],
            "INSTITUTIONAL": []
        }

        for result in all_results:
            by_layer[result["layer"]].append(result)

        return by_layer

    def calculate_shapley_values(self,
                                  context: Dict = None,
                                  n_samples: int = 1000) -> Dict[str, float]:
        """
        Calculate Shapley values for each barrier.

        Shapley value = average marginal contribution across all orderings.
        This gives fair attribution of impact to each barrier.
        """
        barriers = self.model.barrier_set.barriers
        n = len(barriers)
        shapley = {b.name: 0.0 for b in barriers}

        # Monte Carlo approximation of Shapley values
        for _ in range(n_samples):
            # Random permutation
            perm = np.random.permutation(n)

            current_set = []
            prev_value = self.model.calculate_success_probability([], context)

            for idx in perm:
                barrier = barriers[idx]
                current_set.append(barrier)
                new_value = self.model.calculate_success_probability(current_set, context)

                # Marginal contribution (negative because adding barrier reduces success)
                marginal = prev_value - new_value
                shapley[barrier.name] += marginal / n_samples

                prev_value = new_value

        return shapley


# =============================================================================
# PART V: STEPWISE CUMULATIVE REMOVAL ANALYSIS
# =============================================================================

class StepwiseCumulativeAnalysis:
    """
    Analyze effect of removing barriers cumulatively in different orders.

    Strategies:
    - Forward: Remove in natural order (Layer 1 → Layer 2 → Layer 3)
    - Backward: Remove in reverse order
    - Optimal: Remove in order of impact (greedy)
    - Random: Random removal order (for comparison)
    """

    def __init__(self, model: CounterfactualBarrierModel):
        self.model = model

    def _calculate_trajectory(self,
                               removal_order: List[Barrier],
                               context: Dict = None) -> List[Dict]:
        """Calculate success probability trajectory as barriers are removed."""
        trajectory = []
        remaining = list(self.model.barrier_set.barriers)

        # Baseline (all barriers)
        baseline = self.model.calculate_success_probability(remaining, context)
        trajectory.append({
            "step": 0,
            "barriers_removed": [],
            "barriers_remaining": len(remaining),
            "success_probability": baseline,
            "cumulative_change": 0.0
        })

        # Remove barriers one by one
        for i, barrier in enumerate(removal_order):
            if barrier in remaining:
                remaining.remove(barrier)

            new_prob = self.model.calculate_success_probability(remaining, context)

            trajectory.append({
                "step": i + 1,
                "barrier_removed": barrier.name,
                "layer": barrier.layer.name,
                "barriers_remaining": len(remaining),
                "success_probability": new_prob,
                "cumulative_change": new_prob - baseline,
                "step_change": new_prob - trajectory[-1]["success_probability"]
            })

        return trajectory

    def forward_removal(self, context: Dict = None) -> List[Dict]:
        """Remove barriers in natural order (Layer 1 → 2 → 3)."""
        barriers = self.model.barrier_set.barriers
        ordered = sorted(barriers, key=lambda b: (b.layer.value, b.barrier_type.value))
        return self._calculate_trajectory(ordered, context)

    def backward_removal(self, context: Dict = None) -> List[Dict]:
        """Remove barriers in reverse order (Layer 3 → 2 → 1)."""
        barriers = self.model.barrier_set.barriers
        ordered = sorted(barriers, key=lambda b: (-b.layer.value, -b.barrier_type.value))
        return self._calculate_trajectory(ordered, context)

    def optimal_removal(self, context: Dict = None) -> List[Dict]:
        """
        Remove barriers in greedy optimal order.

        At each step, remove the barrier that maximizes success probability.
        """
        remaining = list(self.model.barrier_set.barriers)
        removal_order = []

        while remaining:
            best_barrier = None
            best_improvement = -1
            current_prob = self.model.calculate_success_probability(remaining, context)

            for barrier in remaining:
                test_remaining = [b for b in remaining if b != barrier]
                new_prob = self.model.calculate_success_probability(test_remaining, context)
                improvement = new_prob - current_prob

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_barrier = barrier

            if best_barrier:
                removal_order.append(best_barrier)
                remaining.remove(best_barrier)

        return self._calculate_trajectory(removal_order, context)

    def random_removal(self, context: Dict = None, seed: int = 42) -> List[Dict]:
        """Remove barriers in random order."""
        np.random.seed(seed)
        barriers = list(self.model.barrier_set.barriers)
        np.random.shuffle(barriers)
        return self._calculate_trajectory(barriers, context)

    def cost_optimal_removal(self, context: Dict = None) -> List[Dict]:
        """
        Remove barriers in order of cost-effectiveness.

        Prioritize barriers with high impact per dollar.
        """
        individual = IndividualBarrierAnalysis(self.model)
        results = individual.analyze_all_barriers(context)

        # Sort by cost-effectiveness (impact per dollar)
        results.sort(key=lambda x: x["cost_effectiveness"], reverse=True)

        removal_order = []
        for r in results:
            barrier = self.model.barrier_set.get_barrier_by_type(
                BarrierType[r["barrier_type"]]
            )
            if barrier:
                removal_order.append(barrier)

        return self._calculate_trajectory(removal_order, context)

    def compare_strategies(self, context: Dict = None) -> Dict[str, List[Dict]]:
        """Compare all removal strategies."""
        return {
            "forward": self.forward_removal(context),
            "backward": self.backward_removal(context),
            "optimal": self.optimal_removal(context),
            "cost_optimal": self.cost_optimal_removal(context),
            "random": self.random_removal(context)
        }


# =============================================================================
# PART VI: BARRIER SET/GROUP REMOVAL ANALYSIS
# =============================================================================

class BarrierGroupAnalysis:
    """
    Analyze effect of removing entire groups/layers of barriers.

    Examines:
    - Single layer removal
    - Layer combinations
    - Custom barrier groups
    """

    def __init__(self, model: CounterfactualBarrierModel):
        self.model = model

    def remove_layer(self,
                     layer: BarrierLayer,
                     context: Dict = None) -> Dict:
        """Remove all barriers in a single layer."""
        baseline = self.model.get_baseline(context)

        remaining = [b for b in self.model.barrier_set.barriers
                    if b.layer != layer]
        removed = [b for b in self.model.barrier_set.barriers
                  if b.layer == layer]

        new_rate = self.model.calculate_success_probability(remaining, context)

        return {
            "layer": layer.name,
            "barriers_removed": [b.name for b in removed],
            "n_barriers_removed": len(removed),
            "baseline_success": baseline,
            "new_success_rate": new_rate,
            "absolute_change": new_rate - baseline,
            "relative_change": (new_rate - baseline) / baseline if baseline > 0 else 0,
            "total_removal_cost": sum(b.removal_cost_usd for b in removed),
            "max_removal_time": max(b.removal_time_months for b in removed) if removed else 0
        }

    def remove_layer_combination(self,
                                  layers: List[BarrierLayer],
                                  context: Dict = None) -> Dict:
        """Remove barriers from multiple layers."""
        baseline = self.model.get_baseline(context)

        remaining = [b for b in self.model.barrier_set.barriers
                    if b.layer not in layers]
        removed = [b for b in self.model.barrier_set.barriers
                  if b.layer in layers]

        new_rate = self.model.calculate_success_probability(remaining, context)

        return {
            "layers": [l.name for l in layers],
            "barriers_removed": [b.name for b in removed],
            "n_barriers_removed": len(removed),
            "baseline_success": baseline,
            "new_success_rate": new_rate,
            "absolute_change": new_rate - baseline,
            "relative_change": (new_rate - baseline) / baseline if baseline > 0 else 0,
            "total_removal_cost": sum(b.removal_cost_usd for b in removed),
            "max_removal_time": max(b.removal_time_months for b in removed) if removed else 0
        }

    def analyze_all_layer_combinations(self, context: Dict = None) -> List[Dict]:
        """Analyze all possible layer combinations."""
        layers = list(BarrierLayer)
        results = []

        # Single layers
        for layer in layers:
            results.append(self.remove_layer(layer, context))

        # Layer pairs
        for combo in combinations(layers, 2):
            results.append(self.remove_layer_combination(list(combo), context))

        # All layers
        results.append(self.remove_layer_combination(layers, context))

        # Sort by absolute change
        results.sort(key=lambda x: x["absolute_change"], reverse=True)

        return results

    def calculate_interaction_effects(self, context: Dict = None) -> Dict:
        """
        Calculate interaction effects between layers.

        Interaction = Joint effect - Sum of individual effects

        Positive interaction: Layers reinforce each other
        Negative interaction: Layers are redundant
        """
        # Individual layer effects
        layer_effects = {}
        for layer in BarrierLayer:
            result = self.remove_layer(layer, context)
            layer_effects[layer.name] = result["absolute_change"]

        # Pairwise interactions
        interactions = {}
        for l1, l2 in combinations(BarrierLayer, 2):
            combo_result = self.remove_layer_combination([l1, l2], context)
            joint_effect = combo_result["absolute_change"]
            sum_individual = layer_effects[l1.name] + layer_effects[l2.name]
            interaction = joint_effect - sum_individual

            interactions[f"{l1.name} × {l2.name}"] = {
                "joint_effect": joint_effect,
                "sum_individual": sum_individual,
                "interaction": interaction,
                "interaction_type": "synergistic" if interaction > 0.01 else
                                   "antagonistic" if interaction < -0.01 else "additive"
            }

        # Three-way interaction
        all_layers_result = self.remove_layer_combination(list(BarrierLayer), context)
        sum_all_individual = sum(layer_effects.values())
        three_way = all_layers_result["absolute_change"] - sum_all_individual

        interactions["THREE_WAY"] = {
            "joint_effect": all_layers_result["absolute_change"],
            "sum_individual": sum_all_individual,
            "interaction": three_way,
            "interaction_type": "synergistic" if three_way > 0.01 else
                               "antagonistic" if three_way < -0.01 else "additive"
        }

        return {
            "individual_effects": layer_effects,
            "interactions": interactions
        }


# =============================================================================
# PART VII: COMPREHENSIVE COUNTERFACTUAL ANALYSIS
# =============================================================================

class ComprehensiveCounterfactualAnalysis:
    """
    Unified interface for all counterfactual analyses.
    """

    def __init__(self, barrier_set: BarrierSet = None):
        self.model = CounterfactualBarrierModel(barrier_set)
        self.individual = IndividualBarrierAnalysis(self.model)
        self.stepwise = StepwiseCumulativeAnalysis(self.model)
        self.group = BarrierGroupAnalysis(self.model)

    def full_analysis(self, context: Dict = None) -> Dict:
        """Run complete counterfactual analysis."""
        return {
            "baseline": self.model.get_baseline(context),
            "individual_effects": self.individual.analyze_all_barriers(context),
            "shapley_values": self.individual.calculate_shapley_values(context),
            "stepwise_comparison": self.stepwise.compare_strategies(context),
            "layer_effects": self.group.analyze_all_layer_combinations(context),
            "interactions": self.group.calculate_interaction_effects(context)
        }

    def summary_report(self, context: Dict = None) -> str:
        """Generate human-readable summary report."""
        analysis = self.full_analysis(context)

        lines = [
            "=" * 80,
            "COUNTERFACTUAL BARRIER ANALYSIS SUMMARY",
            "=" * 80,
            "",
            f"Baseline Success Probability: {analysis['baseline']:.1%}",
            f"(With all {len(self.model.barrier_set.barriers)} barriers active)",
            "",
            "-" * 40,
            "TOP 5 INDIVIDUAL BARRIER EFFECTS:",
            "-" * 40,
        ]

        for i, result in enumerate(analysis['individual_effects'][:5], 1):
            lines.append(
                f"  {i}. {result['barrier']}: "
                f"+{result['absolute_change']:.1%} success "
                f"(PAF: {result['paf']:.1%})"
            )

        lines.extend([
            "",
            "-" * 40,
            "LAYER EFFECTS:",
            "-" * 40,
        ])

        for result in analysis['layer_effects'][:3]:
            layers = result.get('layers', [result.get('layer')])
            if isinstance(layers, str):
                layers = [layers]
            lines.append(
                f"  Remove {' + '.join(layers)}: "
                f"+{result['absolute_change']:.1%} success"
            )

        lines.extend([
            "",
            "-" * 40,
            "OPTIMAL REMOVAL STRATEGY:",
            "-" * 40,
        ])

        optimal = analysis['stepwise_comparison']['optimal']
        for step in optimal[1:4]:  # First 3 steps
            lines.append(
                f"  Step {step['step']}: Remove '{step.get('barrier_removed', 'N/A')}' "
                f"→ {step['success_probability']:.1%} success"
            )

        lines.extend([
            "",
            "-" * 40,
            "LAYER INTERACTIONS:",
            "-" * 40,
        ])

        for name, data in analysis['interactions']['interactions'].items():
            lines.append(
                f"  {name}: {data['interaction_type']} "
                f"(interaction: {data['interaction']:+.1%})"
            )

        lines.extend([
            "",
            "=" * 80,
        ])

        return "\n".join(lines)


# =============================================================================
# PART VIII: VISUALIZATION
# =============================================================================

def plot_individual_barrier_effects(analysis: ComprehensiveCounterfactualAnalysis,
                                     context: Dict = None):
    """Plot individual barrier effects."""
    results = analysis.individual.analyze_all_barriers(context)

    names = [r['barrier'][:20] for r in results]
    changes = [r['absolute_change'] * 100 for r in results]
    layers = [r['layer'] for r in results]

    colors = {
        'DATA_INTEGRATION': '#2ecc71',
        'DATA_ACCURACY': '#3498db',
        'INSTITUTIONAL': '#e74c3c'
    }
    bar_colors = [colors[l] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, changes, color=bar_colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Increase in Success Probability (%)', fontsize=11)
    ax.set_title('Individual Barrier Removal Effects\n(Counterfactual Analysis)',
                fontsize=13, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['DATA_INTEGRATION'], label='Data Integration'),
        Patch(facecolor=colors['DATA_ACCURACY'], label='Data Accuracy'),
        Patch(facecolor=colors['INSTITUTIONAL'], label='Institutional')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('individual_barrier_effects.png', dpi=300, bbox_inches='tight')
    print("Saved: individual_barrier_effects.png")
    return fig


def plot_stepwise_comparison(analysis: ComprehensiveCounterfactualAnalysis,
                              context: Dict = None):
    """Plot comparison of stepwise removal strategies."""
    strategies = analysis.stepwise.compare_strategies(context)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        'forward': '#2ecc71',
        'backward': '#e74c3c',
        'optimal': '#9b59b6',
        'cost_optimal': '#f39c12',
        'random': '#95a5a6'
    }

    for name, trajectory in strategies.items():
        steps = [t['step'] for t in trajectory]
        probs = [t['success_probability'] * 100 for t in trajectory]
        ax.plot(steps, probs, 'o-', label=name.replace('_', ' ').title(),
               color=colors[name], linewidth=2, markersize=6)

    ax.set_xlabel('Number of Barriers Removed', fontsize=11)
    ax.set_ylabel('Success Probability (%)', fontsize=11)
    ax.set_title('Stepwise Cumulative Barrier Removal\n(Strategy Comparison)',
                fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, len(strategies['forward']) - 0.5)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('stepwise_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: stepwise_comparison.png")
    return fig


def plot_layer_effects(analysis: ComprehensiveCounterfactualAnalysis,
                        context: Dict = None):
    """Plot layer and combination effects."""
    results = analysis.group.analyze_all_layer_combinations(context)

    # Prepare data
    labels = []
    changes = []
    costs = []

    for r in results:
        if 'layers' in r:
            label = ' + '.join([l.split('_')[0] for l in r['layers']])
        else:
            label = r['layer'].split('_')[0]
        labels.append(label)
        changes.append(r['absolute_change'] * 100)
        costs.append(r['total_removal_cost'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Effect sizes
    ax = axes[0]
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(labels)))
    bars = ax.bar(range(len(labels)), changes, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Increase in Success Probability (%)', fontsize=11)
    ax.set_title('Effect of Removing Barrier Layers/Groups',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: Cost vs Effect
    ax = axes[1]
    scatter = ax.scatter(costs, changes, c=range(len(labels)),
                        cmap='viridis', s=100, alpha=0.7)
    for i, label in enumerate(labels):
        ax.annotate(label, (costs[i], changes[i]), fontsize=8,
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Total Removal Cost (USD)', fontsize=11)
    ax.set_ylabel('Increase in Success Probability (%)', fontsize=11)
    ax.set_title('Cost-Effectiveness of Layer Removal',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('layer_effects.png', dpi=300, bbox_inches='tight')
    print("Saved: layer_effects.png")
    return fig


def plot_interaction_heatmap(analysis: ComprehensiveCounterfactualAnalysis,
                              context: Dict = None):
    """Plot interaction effects as heatmap."""
    interactions = analysis.group.calculate_interaction_effects(context)

    layers = ['DATA_INTEGRATION', 'DATA_ACCURACY', 'INSTITUTIONAL']
    n = len(layers)

    # Create matrix
    matrix = np.zeros((n, n))

    # Fill diagonal with individual effects
    for i, layer in enumerate(layers):
        matrix[i, i] = interactions['individual_effects'][layer]

    # Fill off-diagonal with interactions
    for i, l1 in enumerate(layers):
        for j, l2 in enumerate(layers):
            if i < j:
                key = f"{l1} × {l2}"
                if key in interactions['interactions']:
                    matrix[i, j] = interactions['interactions'][key]['interaction']
                    matrix[j, i] = matrix[i, j]

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                   vmin=-0.2, vmax=0.2)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [l.replace('_', '\n') for l in layers]
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Add values
    for i in range(n):
        for j in range(n):
            text = f"{matrix[i,j]:.1%}"
            color = 'white' if abs(matrix[i,j]) > 0.1 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=11)

    ax.set_title('Layer Effects & Interactions\n(Diagonal: Individual, Off-diagonal: Interaction)',
                fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Effect Size', fontsize=10)

    plt.tight_layout()
    plt.savefig('interaction_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: interaction_heatmap.png")
    return fig


# =============================================================================
# PART IX: MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("COUNTERFACTUAL BARRIER ANALYSIS FOR ALGORITHMIC BIAS")
    print("Individual | Stepwise Cumulative | Barrier Group Removal")
    print("=" * 80)

    # Initialize analysis
    analysis = ComprehensiveCounterfactualAnalysis()

    # Print summary report
    print(analysis.summary_report())

    # Detailed individual analysis
    print("\n" + "=" * 80)
    print("DETAILED INDIVIDUAL BARRIER EFFECTS")
    print("=" * 80)

    individual_results = analysis.individual.analyze_all_barriers()
    print(f"\n{'Barrier':<30} {'Change':>10} {'PAF':>10} {'Cost':>10} {'Layer':<20}")
    print("-" * 85)

    for r in individual_results:
        print(f"{r['barrier']:<30} {r['absolute_change']*100:>9.1f}% "
              f"{r['paf']*100:>9.1f}% ${r['removal_cost']:>8.0f} {r['layer']:<20}")

    # Shapley values
    print("\n" + "=" * 80)
    print("SHAPLEY VALUES (Fair Attribution)")
    print("=" * 80)

    shapley = analysis.individual.calculate_shapley_values()
    sorted_shapley = sorted(shapley.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Barrier':<30} {'Shapley Value':>15}")
    print("-" * 50)
    for name, value in sorted_shapley:
        print(f"{name:<30} {value*100:>14.2f}%")

    # Stepwise comparison
    print("\n" + "=" * 80)
    print("STEPWISE REMOVAL TRAJECTORIES")
    print("=" * 80)

    strategies = analysis.stepwise.compare_strategies()

    print("\nFinal success rates by strategy:")
    for name, trajectory in strategies.items():
        final = trajectory[-1]['success_probability']
        print(f"  {name.replace('_', ' ').title():<15}: {final:.1%}")

    # Layer interactions
    print("\n" + "=" * 80)
    print("LAYER INTERACTION EFFECTS")
    print("=" * 80)

    interactions = analysis.group.calculate_interaction_effects()

    print("\nIndividual layer effects:")
    for layer, effect in interactions['individual_effects'].items():
        print(f"  {layer}: +{effect:.1%}")

    print("\nInteraction effects:")
    for name, data in interactions['interactions'].items():
        print(f"  {name}: {data['interaction_type']} ({data['interaction']:+.1%})")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    try:
        plot_individual_barrier_effects(analysis)
        plot_stepwise_comparison(analysis)
        plot_layer_effects(analysis)
        plot_interaction_heatmap(analysis)
        print("\nAll visualizations saved successfully.")
    except Exception as e:
        print(f"Visualization error: {e}")

    print("\n" + "=" * 80)
    print("COUNTERFACTUAL ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
