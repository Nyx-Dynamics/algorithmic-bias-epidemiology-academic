"""
BARRIER REMOVAL VISUALIZATION AND DATA EXPORT
Algorithmic Bias Epidemiology Framework

Generates publication-quality figures and CSV exports for:
1. Individual barrier removal effects
2. Stepwise cumulative removal strategies
3. Layer interaction heatmaps
4. Cost-effectiveness analysis

Author: AC Demidont, DO
Nyx Dynamics LLC
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
import itertools
import math


@dataclass
class Barrier:
    """Individual barrier in the algorithmic bias system."""
    name: str
    layer: str  # 'data_integration', 'data_accuracy', 'institutional'
    base_probability: float  # Probability of passing this barrier
    removal_cost: float  # Estimated cost to remove ($)
    description: str


class BarrierRemovalModel:
    """
    Counterfactual barrier removal analysis.

    Models the effect of removing barriers individually, stepwise,
    and in combination to understand the synergistic structure
    of the algorithmic bias barrier system.
    """

    def __init__(self):
        self.barriers = self._define_barriers()
        self.layers = ['data_integration', 'data_accuracy', 'institutional']

    def _define_barriers(self) -> Dict[str, Barrier]:
        """Define the 11-barrier model."""
        return {
            # Layer 1: Data Integration Barriers
            'rapid_transmission': Barrier(
                name='Rapid Data Transmission',
                layer='data_integration',
                base_probability=0.30,
                removal_cost=500,
                description='Data spreads to CRAs within days'
            ),
            'multi_system_integration': Barrier(
                name='Multi-System Integration',
                layer='data_integration',
                base_probability=0.55,
                removal_cost=1000,
                description='Data copied across 20+ databases'
            ),
            'permanent_storage': Barrier(
                name='Permanent Storage',
                layer='data_integration',
                base_probability=0.45,
                removal_cost=2000,
                description='No automatic expiration of records'
            ),

            # Layer 2: Data Accuracy Barriers
            'error_detection': Barrier(
                name='Error Detection Difficulty',
                layer='data_accuracy',
                base_probability=0.35,
                removal_cost=300,
                description='Errors difficult to identify'
            ),
            'correction_process': Barrier(
                name='Correction Process Barriers',
                layer='data_accuracy',
                base_probability=0.35,
                removal_cost=800,
                description='Complex, slow correction procedures'
            ),
            'incomplete_propagation': Barrier(
                name='Incomplete Correction Propagation',
                layer='data_accuracy',
                base_probability=0.40,
                removal_cost=1500,
                description='Corrections not applied across systems'
            ),

            # Layer 3: Institutional Barriers
            'awareness_gap': Barrier(
                name='Awareness Gap',
                layer='institutional',
                base_probability=0.30,
                removal_cost=100,
                description='Individuals unaware of adverse data'
            ),
            'record_access': Barrier(
                name='Record Access Barriers',
                layer='institutional',
                base_probability=0.55,
                removal_cost=200,
                description='Difficulty obtaining own records'
            ),
            'legal_knowledge': Barrier(
                name='Legal Knowledge Gap',
                layer='institutional',
                base_probability=0.25,
                removal_cost=500,
                description='Not knowing legal rights/remedies'
            ),
            'legal_resources': Barrier(
                name='Legal Resource Barriers',
                layer='institutional',
                base_probability=0.40,
                removal_cost=5000,
                description='Cost of legal representation'
            ),
            'systemic_bias': Barrier(
                name='Systemic Bias in Algorithms',
                layer='institutional',
                base_probability=0.30,
                removal_cost=10000,
                description='Bias encoded in algorithm design'
            ),
        }

    def calculate_baseline_success(self) -> float:
        """
        Calculate baseline success probability with all barriers present.

        Uses multiplicative model: P(success) = Product of all barrier probabilities
        """
        prob = 1.0
        for barrier in self.barriers.values():
            prob *= barrier.base_probability
        return prob

    def calculate_success_removing_barrier(self, barrier_key: str) -> float:
        """Calculate success probability when one barrier is removed (set to 1.0)."""
        prob = 1.0
        for key, barrier in self.barriers.items():
            if key == barrier_key:
                prob *= 1.0  # Barrier removed
            else:
                prob *= barrier.base_probability
        return prob

    def calculate_success_removing_barriers(self, barrier_keys: List[str]) -> float:
        """Calculate success probability when multiple barriers are removed."""
        prob = 1.0
        for key, barrier in self.barriers.items():
            if key in barrier_keys:
                prob *= 1.0  # Barrier removed
            else:
                prob *= barrier.base_probability
        return prob

    def individual_barrier_effects(self) -> pd.DataFrame:
        """
        Calculate marginal effect of removing each barrier individually.

        Returns DataFrame with barrier effects and metadata.
        """
        baseline = self.calculate_baseline_success()

        results = []
        for key, barrier in self.barriers.items():
            new_success = self.calculate_success_removing_barrier(key)
            effect = new_success - baseline

            results.append({
                'barrier_key': key,
                'barrier_name': barrier.name,
                'layer': barrier.layer,
                'base_probability': barrier.base_probability,
                'removal_cost': barrier.removal_cost,
                'baseline_success': baseline,
                'success_after_removal': new_success,
                'marginal_effect': effect,
                'marginal_effect_pct': effect * 100,
                'cost_per_pct_improvement': barrier.removal_cost / (effect * 100) if effect > 0 else np.inf,
            })

        return pd.DataFrame(results)

    def stepwise_removal_strategies(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate stepwise cumulative barrier removal for different strategies.

        Strategies:
        - forward: Layer 1 → 2 → 3
        - backward: Layer 3 → 2 → 1
        - optimal: Greedy by impact
        - cost_optimal: Greedy by cost-effectiveness
        - random: Random order (average of 10 runs)
        """
        strategies = {}

        # Forward: Layer 1 → 2 → 3
        forward_order = (
            [k for k, b in self.barriers.items() if b.layer == 'data_integration'] +
            [k for k, b in self.barriers.items() if b.layer == 'data_accuracy'] +
            [k for k, b in self.barriers.items() if b.layer == 'institutional']
        )
        strategies['forward'] = self._stepwise_removal(forward_order)

        # Backward: Layer 3 → 2 → 1
        backward_order = (
            [k for k, b in self.barriers.items() if b.layer == 'institutional'] +
            [k for k, b in self.barriers.items() if b.layer == 'data_accuracy'] +
            [k for k, b in self.barriers.items() if b.layer == 'data_integration']
        )
        strategies['backward'] = self._stepwise_removal(backward_order)

        # Optimal: Greedy by impact
        strategies['optimal'] = self._greedy_removal_by_impact()

        # Cost-optimal: Greedy by cost-effectiveness
        strategies['cost_optimal'] = self._greedy_removal_by_cost()

        # Random: Average of multiple runs
        random_results = []
        for _ in range(10):
            order = list(self.barriers.keys())
            np.random.shuffle(order)
            random_results.append(self._stepwise_removal(order))

        # Average random runs
        avg_random = random_results[0].copy()
        for col in ['success_probability', 'cumulative_cost']:
            values = np.mean([r[col].values for r in random_results], axis=0)
            avg_random[col] = values
        strategies['random'] = avg_random

        return strategies

    def _stepwise_removal(self, order: List[str]) -> pd.DataFrame:
        """Calculate stepwise removal following given order."""
        results = []
        removed = []
        cumulative_cost = 0

        # Baseline (no barriers removed)
        results.append({
            'step': 0,
            'barriers_removed': 0,
            'barrier_key': 'baseline',
            'barrier_name': 'Baseline (all barriers)',
            'success_probability': self.calculate_baseline_success(),
            'cumulative_cost': 0,
        })

        for i, key in enumerate(order):
            removed.append(key)
            barrier = self.barriers[key]
            cumulative_cost += barrier.removal_cost

            results.append({
                'step': i + 1,
                'barriers_removed': i + 1,
                'barrier_key': key,
                'barrier_name': barrier.name,
                'success_probability': self.calculate_success_removing_barriers(removed),
                'cumulative_cost': cumulative_cost,
            })

        return pd.DataFrame(results)

    def _greedy_removal_by_impact(self) -> pd.DataFrame:
        """Greedy removal selecting highest impact barrier at each step."""
        results = []
        removed = []
        remaining = list(self.barriers.keys())

        # Baseline
        results.append({
            'step': 0,
            'barriers_removed': 0,
            'barrier_key': 'baseline',
            'barrier_name': 'Baseline (all barriers)',
            'success_probability': self.calculate_baseline_success(),
            'cumulative_cost': 0,
        })

        cumulative_cost = 0

        while remaining:
            # Find barrier with highest marginal impact
            best_key = None
            best_impact = -1
            current_success = self.calculate_success_removing_barriers(removed)

            for key in remaining:
                test_removed = removed + [key]
                new_success = self.calculate_success_removing_barriers(test_removed)
                impact = new_success - current_success

                if impact > best_impact:
                    best_impact = impact
                    best_key = key

            removed.append(best_key)
            remaining.remove(best_key)
            cumulative_cost += self.barriers[best_key].removal_cost

            results.append({
                'step': len(removed),
                'barriers_removed': len(removed),
                'barrier_key': best_key,
                'barrier_name': self.barriers[best_key].name,
                'success_probability': self.calculate_success_removing_barriers(removed),
                'cumulative_cost': cumulative_cost,
            })

        return pd.DataFrame(results)

    def _greedy_removal_by_cost(self) -> pd.DataFrame:
        """Greedy removal by cost-effectiveness (impact per dollar)."""
        results = []
        removed = []
        remaining = list(self.barriers.keys())

        # Baseline
        results.append({
            'step': 0,
            'barriers_removed': 0,
            'barrier_key': 'baseline',
            'barrier_name': 'Baseline (all barriers)',
            'success_probability': self.calculate_baseline_success(),
            'cumulative_cost': 0,
        })

        cumulative_cost = 0

        while remaining:
            # Find barrier with best cost-effectiveness
            best_key = None
            best_ratio = -1
            current_success = self.calculate_success_removing_barriers(removed)

            for key in remaining:
                test_removed = removed + [key]
                new_success = self.calculate_success_removing_barriers(test_removed)
                impact = new_success - current_success
                cost = self.barriers[key].removal_cost

                ratio = impact / cost if cost > 0 else 0

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_key = key

            removed.append(best_key)
            remaining.remove(best_key)
            cumulative_cost += self.barriers[best_key].removal_cost

            results.append({
                'step': len(removed),
                'barriers_removed': len(removed),
                'barrier_key': best_key,
                'barrier_name': self.barriers[best_key].name,
                'success_probability': self.calculate_success_removing_barriers(removed),
                'cumulative_cost': cumulative_cost,
            })

        return pd.DataFrame(results)

    def layer_removal_effects(self) -> pd.DataFrame:
        """Calculate effect of removing entire layers and combinations."""
        results = []

        layer_barriers = {
            layer: [k for k, b in self.barriers.items() if b.layer == layer]
            for layer in self.layers
        }

        baseline = self.calculate_baseline_success()

        # Single layers
        for layer in self.layers:
            keys = layer_barriers[layer]
            success = self.calculate_success_removing_barriers(keys)
            cost = sum(self.barriers[k].removal_cost for k in keys)

            results.append({
                'combination': layer,
                'layers_removed': 1,
                'barriers_removed': len(keys),
                'success_probability': success,
                'effect': success - baseline,
                'effect_pct': (success - baseline) * 100,
                'total_cost': cost,
            })

        # Pairs of layers
        for combo in itertools.combinations(self.layers, 2):
            keys = layer_barriers[combo[0]] + layer_barriers[combo[1]]
            success = self.calculate_success_removing_barriers(keys)
            cost = sum(self.barriers[k].removal_cost for k in keys)

            results.append({
                'combination': ' + '.join(combo),
                'layers_removed': 2,
                'barriers_removed': len(keys),
                'success_probability': success,
                'effect': success - baseline,
                'effect_pct': (success - baseline) * 100,
                'total_cost': cost,
            })

        # All three layers
        all_keys = list(self.barriers.keys())
        success = self.calculate_success_removing_barriers(all_keys)
        cost = sum(b.removal_cost for b in self.barriers.values())

        results.append({
            'combination': 'ALL LAYERS',
            'layers_removed': 3,
            'barriers_removed': len(all_keys),
            'success_probability': success,
            'effect': success - baseline,
            'effect_pct': (success - baseline) * 100,
            'total_cost': cost,
        })

        return pd.DataFrame(results)

    def interaction_effects(self) -> pd.DataFrame:
        """Calculate interaction effects between layers (synergy analysis)."""
        layer_barriers = {
            layer: [k for k, b in self.barriers.items() if b.layer == layer]
            for layer in self.layers
        }

        baseline = self.calculate_baseline_success()

        # Individual layer effects
        individual_effects = {}
        for layer in self.layers:
            keys = layer_barriers[layer]
            success = self.calculate_success_removing_barriers(keys)
            individual_effects[layer] = success - baseline

        # Pairwise effects
        pairwise_effects = {}
        for combo in itertools.combinations(self.layers, 2):
            keys = layer_barriers[combo[0]] + layer_barriers[combo[1]]
            success = self.calculate_success_removing_barriers(keys)
            joint_effect = success - baseline

            # Interaction = joint - sum of individuals
            expected = individual_effects[combo[0]] + individual_effects[combo[1]]
            interaction = joint_effect - expected

            pairwise_effects[combo] = {
                'joint_effect': joint_effect,
                'expected_additive': expected,
                'interaction': interaction,
            }

        # Three-way interaction
        all_keys = list(self.barriers.keys())
        all_success = self.calculate_success_removing_barriers(all_keys)
        all_effect = all_success - baseline

        # Expected from lower-order terms
        sum_individual = sum(individual_effects.values())
        sum_pairwise = sum(p['interaction'] for p in pairwise_effects.values())
        expected_three = sum_individual + sum_pairwise

        three_way_interaction = all_effect - expected_three

        # Build results DataFrame
        results = []

        for layer in self.layers:
            results.append({
                'type': 'individual',
                'term': layer,
                'effect': individual_effects[layer],
                'effect_pct': individual_effects[layer] * 100,
            })

        for combo, data in pairwise_effects.items():
            results.append({
                'type': 'pairwise',
                'term': ' x '.join(combo),
                'effect': data['interaction'],
                'effect_pct': data['interaction'] * 100,
            })

        results.append({
            'type': 'three-way',
            'term': ' x '.join(self.layers),
            'effect': three_way_interaction,
            'effect_pct': three_way_interaction * 100,
        })

        return pd.DataFrame(results)

    def shapley_values(self) -> pd.DataFrame:
        """
        Calculate Shapley values for barrier attribution.

        Shapley value gives the fair attribution of the total effect
        to each barrier, accounting for all possible orderings.
        """
        all_keys = list(self.barriers.keys())
        n = len(all_keys)
        baseline = self.calculate_baseline_success()
        full_effect = self.calculate_success_removing_barriers(all_keys) - baseline

        shapley = {key: 0.0 for key in all_keys}

        # For computational tractability, sample orderings
        n_samples = min(1000, math.factorial(n))

        for _ in range(n_samples):
            order = list(all_keys)
            np.random.shuffle(order)

            removed = []
            prev_success = baseline

            for key in order:
                removed.append(key)
                new_success = self.calculate_success_removing_barriers(removed)
                marginal = new_success - prev_success
                shapley[key] += marginal / n_samples
                prev_success = new_success

        results = []
        for key, value in shapley.items():
            barrier = self.barriers[key]
            results.append({
                'barrier_key': key,
                'barrier_name': barrier.name,
                'layer': barrier.layer,
                'shapley_value': value,
                'shapley_pct': value * 100,
                'attribution_pct': (value / full_effect * 100) if full_effect > 0 else 0,
            })

        return pd.DataFrame(results).sort_values('shapley_value', ascending=False)


def generate_visualizations(model: BarrierRemovalModel, output_dir: str = '.'):
    """Generate all visualizations for barrier removal analysis."""

    # 1. Individual barrier effects
    individual_df = model.individual_barrier_effects()

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {'data_integration': '#2ecc71', 'data_accuracy': '#3498db', 'institutional': '#e74c3c'}

    bars = ax.barh(
        range(len(individual_df)),
        individual_df['marginal_effect_pct'],
        color=[colors[l] for l in individual_df['layer']]
    )

    ax.set_yticks(range(len(individual_df)))
    ax.set_yticklabels(individual_df['barrier_name'])
    ax.set_xlabel('Marginal Effect (%)', fontsize=12)
    ax.set_title('Individual Barrier Removal Effects\n(All effects near 0% due to multiplicative blocking)',
                fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Data Integration'),
        Patch(facecolor='#3498db', label='Data Accuracy'),
        Patch(facecolor='#e74c3c', label='Institutional'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/individual_barrier_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Stepwise removal strategies
    strategies = model.stepwise_removal_strategies()

    fig, ax = plt.subplots(figsize=(12, 8))

    colors_strat = {
        'forward': '#2ecc71',
        'backward': '#e74c3c',
        'optimal': '#9b59b6',
        'cost_optimal': '#f39c12',
        'random': '#95a5a6',
    }

    for name, df in strategies.items():
        ax.plot(df['barriers_removed'], df['success_probability'] * 100,
               'o-', label=name.replace('_', ' ').title(),
               color=colors_strat[name], linewidth=2, markersize=6)

    ax.set_xlabel('Number of Barriers Removed', fontsize=12)
    ax.set_ylabel('Success Probability (%)', fontsize=12)
    ax.set_title('Stepwise Barrier Removal: Strategy Comparison\n(All strategies converge only at complete removal)',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/stepwise_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Layer removal effects
    layer_df = model.layer_removal_effects()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Effect by layer combination
    ax = axes[0]
    bars = ax.bar(range(len(layer_df)), layer_df['effect_pct'],
                 color=plt.cm.Greens(np.linspace(0.3, 0.9, len(layer_df))))

    ax.set_xticks(range(len(layer_df)))
    ax.set_xticklabels(layer_df['combination'], rotation=45, ha='right')
    ax.set_ylabel('Effect (%)', fontsize=12)
    ax.set_title('A. Effect of Layer Removal', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: Cost-effectiveness
    ax = axes[1]
    ax.scatter(layer_df['total_cost'], layer_df['effect_pct'],
              s=layer_df['barriers_removed'] * 100,
              c=layer_df['effect_pct'], cmap='RdYlGn',
              alpha=0.7, edgecolors='black')

    for i, row in layer_df.iterrows():
        ax.annotate(row['combination'], (row['total_cost'], row['effect_pct']),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Total Cost ($)', fontsize=12)
    ax.set_ylabel('Effect (%)', fontsize=12)
    ax.set_title('B. Cost-Effectiveness of Layer Removal', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/layer_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Interaction heatmap
    interaction_df = model.interaction_effects()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create matrix for heatmap
    layers = model.layers
    matrix = np.zeros((3, 3))

    # Fill diagonal with individual effects
    for i, layer in enumerate(layers):
        row = interaction_df[interaction_df['term'] == layer]
        if len(row) > 0:
            matrix[i, i] = row['effect_pct'].values[0]

    # Fill off-diagonal with pairwise interactions
    for combo in itertools.combinations(range(3), 2):
        term = f'{layers[combo[0]]} x {layers[combo[1]]}'
        row = interaction_df[interaction_df['term'] == term]
        if len(row) > 0:
            matrix[combo[0], combo[1]] = row['effect_pct'].values[0]
            matrix[combo[1], combo[0]] = row['effect_pct'].values[0]

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=-5, vmax=10)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    labels = ['Data\nIntegration', 'Data\nAccuracy', 'Institutional']
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Add value annotations
    for i in range(3):
        for j in range(3):
            text = f'{matrix[i, j]:.1f}%'
            ax.text(j, i, text, ha='center', va='center', fontsize=12,
                   color='white' if abs(matrix[i, j]) > 3 else 'black')

    ax.set_title('Layer Interaction Effects Matrix\n(Diagonal: Individual, Off-diagonal: Pairwise)',
                fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Effect (%)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/interaction_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Shapley values
    shapley_df = model.shapley_values()

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {'data_integration': '#2ecc71', 'data_accuracy': '#3498db', 'institutional': '#e74c3c'}
    bar_colors = [colors[l] for l in shapley_df['layer']]

    bars = ax.barh(range(len(shapley_df)), shapley_df['attribution_pct'], color=bar_colors)

    ax.set_yticks(range(len(shapley_df)))
    ax.set_yticklabels(shapley_df['barrier_name'])
    ax.set_xlabel('Attribution (%)', fontsize=12)
    ax.set_title('Shapley Value Attribution of Success Improvement\n(Fair allocation accounting for all orderings)',
                fontsize=14, fontweight='bold')

    # Legend
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Data Integration'),
        Patch(facecolor='#3498db', label='Data Accuracy'),
        Patch(facecolor='#e74c3c', label='Institutional'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/shapley_attribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to {output_dir}/")


def export_csv_data(model: BarrierRemovalModel, output_dir: str = '.'):
    """Export all analysis results to CSV files."""

    # 1. Individual barrier effects
    individual_df = model.individual_barrier_effects()
    individual_df.to_csv(f'{output_dir}/individual_barrier_effects.csv', index=False)

    # 2. Stepwise removal strategies
    strategies = model.stepwise_removal_strategies()
    for name, df in strategies.items():
        df.to_csv(f'{output_dir}/stepwise_{name}.csv', index=False)

    # Combined stepwise for comparison
    combined = pd.DataFrame()
    for name, df in strategies.items():
        df_copy = df[['barriers_removed', 'success_probability']].copy()
        df_copy.columns = ['barriers_removed', f'success_{name}']
        if combined.empty:
            combined = df_copy
        else:
            combined = combined.merge(df_copy, on='barriers_removed')
    combined.to_csv(f'{output_dir}/stepwise_all_strategies.csv', index=False)

    # 3. Layer removal effects
    layer_df = model.layer_removal_effects()
    layer_df.to_csv(f'{output_dir}/layer_removal_effects.csv', index=False)

    # 4. Interaction effects
    interaction_df = model.interaction_effects()
    interaction_df.to_csv(f'{output_dir}/interaction_effects.csv', index=False)

    # 5. Shapley values
    shapley_df = model.shapley_values()
    shapley_df.to_csv(f'{output_dir}/shapley_values.csv', index=False)

    # 6. Barrier definitions
    barrier_df = pd.DataFrame([
        {
            'key': k,
            'name': b.name,
            'layer': b.layer,
            'base_probability': b.base_probability,
            'removal_cost': b.removal_cost,
            'description': b.description,
        }
        for k, b in model.barriers.items()
    ])
    barrier_df.to_csv(f'{output_dir}/barrier_definitions.csv', index=False)

    print(f"CSV files exported to {output_dir}/")


def main():
    """Run barrier removal analysis and generate outputs."""
    print("=" * 80)
    print("BARRIER REMOVAL VISUALIZATION AND DATA EXPORT")
    print("Algorithmic Bias Epidemiology")
    print("=" * 80)

    model = BarrierRemovalModel()

    # Summary statistics
    baseline = model.calculate_baseline_success()
    full_success = model.calculate_success_removing_barriers(list(model.barriers.keys()))

    print(f"\nBaseline success probability: {baseline*100:.4f}%")
    print(f"Success with all barriers removed: {full_success*100:.1f}%")
    print(f"Maximum improvement possible: {(full_success - baseline)*100:.1f}%")

    # Generate outputs
    generate_visualizations(model)
    export_csv_data(model)

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    interaction_df = model.interaction_effects()
    three_way = interaction_df[interaction_df['type'] == 'three-way']['effect_pct'].values[0]

    print(f"\n1. Three-way interaction effect: {three_way:.1f}%")
    print("   → Barrier synergy explains why piecemeal reform fails")

    shapley_df = model.shapley_values()
    top_3 = shapley_df.head(3)
    print("\n2. Top 3 barriers by Shapley attribution:")
    for _, row in top_3.iterrows():
        print(f"   - {row['barrier_name']}: {row['attribution_pct']:.1f}%")

    print("\n3. All removal strategies converge only at complete barrier removal")
    print("   → Partial reform is mathematically futile")


if __name__ == "__main__":
    main()
