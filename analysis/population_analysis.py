"""
POPULATION ATTRIBUTABLE FRACTION (PAF) ANALYSIS
For Algorithmic Bias Epidemiology Framework

Calculates the proportion of adverse outcomes attributable to algorithmic
discrimination across different population groups.

PAF Formula:
    PAF = P_exposed * (RR - 1) / [P_exposed * (RR - 1) + 1]

Where:
    P_exposed = Prevalence of algorithmic bias exposure in the population
    RR = Relative risk of adverse outcome given exposure

Author: AC Demidont, DO
Nyx Dynamics LLC
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json


@dataclass
class PopulationGroup:
    """Defines a population group for PAF analysis."""
    name: str
    prevalence_exposed: float  # P(exposed to algorithmic bias)
    relative_risk: float  # RR for adverse outcomes
    population_size: int  # Estimated US population
    description: str


class PopulationAttributableFractionAnalysis:
    """
    Population Attributable Fraction analysis for algorithmic bias.

    Parallels epidemiological studies of disease burden attribution.
    """

    def __init__(self):
        self.populations = self._define_populations()

    def _define_populations(self) -> Dict[str, PopulationGroup]:
        """Define population groups with exposure and risk data."""
        return {
            'general': PopulationGroup(
                name='General Population',
                prevalence_exposed=0.40,
                relative_risk=1.5,
                population_size=260_000_000,  # US adults
                description='Adults who have interacted with algorithmic screening'
            ),
            'pwid': PopulationGroup(
                name='People Who Inject Drugs (PWID)',
                prevalence_exposed=0.85,
                relative_risk=3.2,
                population_size=3_700_000,
                description='Higher exposure through criminal justice, healthcare, employment'
            ),
            'pwh': PopulationGroup(
                name='People With HIV (PWH)',
                prevalence_exposed=0.70,
                relative_risk=2.4,
                population_size=1_200_000,
                description='Algorithmic exclusion from clinical trials, insurance, employment'
            ),
            'justice_involved': PopulationGroup(
                name='Justice-Involved',
                prevalence_exposed=0.90,
                relative_risk=4.1,
                population_size=19_000_000,  # Ever incarcerated
                description='Criminal records in background checks, employment screening'
            ),
            'formerly_unhoused': PopulationGroup(
                name='Formerly Unhoused',
                prevalence_exposed=0.80,
                relative_risk=2.8,
                population_size=3_500_000,
                description='Housing history in tenant screening algorithms'
            ),
            'black_americans': PopulationGroup(
                name='Black Americans',
                prevalence_exposed=0.55,
                relative_risk=2.1,
                population_size=47_000_000,
                description='Documented bias in credit, employment, healthcare algorithms'
            ),
            'disabled': PopulationGroup(
                name='People with Disabilities',
                prevalence_exposed=0.60,
                relative_risk=1.9,
                population_size=42_000_000,
                description='Employment screening, insurance algorithms'
            ),
        }

    def calculate_paf(self, population_key: str) -> float:
        """
        Calculate Population Attributable Fraction.

        PAF = P_e * (RR - 1) / [P_e * (RR - 1) + 1]

        Interpretation: Proportion of adverse outcomes that would be
        prevented if algorithmic bias exposure were eliminated.
        """
        pop = self.populations[population_key]
        p_e = pop.prevalence_exposed
        rr = pop.relative_risk

        numerator = p_e * (rr - 1)
        denominator = numerator + 1

        return numerator / denominator

    def calculate_all_paf(self) -> Dict[str, float]:
        """Calculate PAF for all population groups."""
        return {key: self.calculate_paf(key) for key in self.populations}

    def calculate_attributable_cases(self,
                                     population_key: str,
                                     baseline_adverse_rate: float = 0.20) -> Dict:
        """
        Calculate the number of adverse outcomes attributable to algorithmic bias.

        Args:
            population_key: Key for population group
            baseline_adverse_rate: Background rate of adverse outcomes (e.g., 20%)

        Returns:
            Dictionary with attributable cases and prevented cases under elimination
        """
        pop = self.populations[population_key]
        paf = self.calculate_paf(population_key)

        # Total adverse outcomes in population
        total_adverse = int(pop.population_size * baseline_adverse_rate)

        # Attributable to algorithmic bias
        attributable = int(total_adverse * paf)

        return {
            'population': pop.name,
            'population_size': pop.population_size,
            'paf': paf,
            'total_adverse_outcomes': total_adverse,
            'attributable_to_algorithms': attributable,
            'potentially_preventable': attributable,
        }

    def calculate_national_burden(self, baseline_rate: float = 0.20) -> Dict:
        """
        Calculate total national burden of algorithmic discrimination.

        Returns aggregate statistics across all defined populations.
        """
        results = {}
        total_attributable = 0
        total_population = 0

        for key in self.populations:
            pop = self.populations[key]
            cases = self.calculate_attributable_cases(key, baseline_rate)
            results[key] = cases

            # Avoid double-counting (general population overlaps with others)
            if key != 'general':
                total_attributable += cases['attributable_to_algorithms']
                total_population += pop.population_size

        # Estimate for general population (non-overlapping)
        general_only = self.populations['general'].population_size - total_population
        if general_only > 0:
            general_paf = self.calculate_paf('general')
            general_attributable = int(general_only * baseline_rate * general_paf)
            total_attributable += general_attributable

        return {
            'by_population': results,
            'total_attributable_cases': total_attributable,
            'methodology': 'Population Attributable Fraction with Levin formula'
        }

    def sensitivity_analysis(self,
                            population_key: str,
                            prevalence_range: Tuple[float, float] = (0.1, 0.9),
                            rr_range: Tuple[float, float] = (1.1, 5.0),
                            n_points: int = 50) -> Dict:
        """
        Sensitivity analysis varying prevalence and relative risk.

        Creates a grid of PAF values to show robustness of findings.
        """
        prevalences = np.linspace(prevalence_range[0], prevalence_range[1], n_points)
        rrs = np.linspace(rr_range[0], rr_range[1], n_points)

        paf_grid = np.zeros((n_points, n_points))

        for i, p in enumerate(prevalences):
            for j, rr in enumerate(rrs):
                numerator = p * (rr - 1)
                paf_grid[i, j] = numerator / (numerator + 1)

        # Point estimate
        pop = self.populations[population_key]
        point_estimate = self.calculate_paf(population_key)

        return {
            'prevalences': prevalences,
            'relative_risks': rrs,
            'paf_grid': paf_grid,
            'point_estimate': point_estimate,
            'point_prevalence': pop.prevalence_exposed,
            'point_rr': pop.relative_risk,
        }

    def plot_paf_comparison(self, save_path: str = None):
        """Create bar chart comparing PAF across populations."""
        paf_values = self.calculate_all_paf()

        # Sort by PAF
        sorted_items = sorted(paf_values.items(), key=lambda x: x[1], reverse=True)
        names = [self.populations[k].name for k, v in sorted_items]
        values = [v * 100 for k, v in sorted_items]

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(names)))
        bars = ax.barh(range(len(names)), values, color=colors)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Population Attributable Fraction (%)', fontsize=12)
        ax.set_title('Proportion of Adverse Outcomes Attributable to Algorithmic Bias\nby Population Group',
                    fontsize=14, fontweight='bold')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10)

        ax.set_xlim(0, 85)
        ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig

    def plot_sensitivity_heatmap(self, population_key: str, save_path: str = None):
        """Create sensitivity analysis heatmap."""
        sens = self.sensitivity_analysis(population_key)

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(sens['paf_grid'] * 100, cmap='YlOrRd', aspect='auto',
                      origin='lower', vmin=0, vmax=80)

        # Mark point estimate
        pop = self.populations[population_key]
        p_idx = int((pop.prevalence_exposed - 0.1) / 0.8 * 49)
        rr_idx = int((pop.relative_risk - 1.1) / 3.9 * 49)
        ax.plot(rr_idx, p_idx, 'ko', markersize=15, markerfacecolor='none',
               markeredgewidth=2, label='Point estimate')

        # Labels
        ax.set_xlabel('Relative Risk (RR)', fontsize=12)
        ax.set_ylabel('Prevalence of Exposure', fontsize=12)
        ax.set_title(f'PAF Sensitivity Analysis: {pop.name}\nPoint Estimate: {sens["point_estimate"]*100:.1f}%',
                    fontsize=12, fontweight='bold')

        # Ticks
        xticks = np.linspace(0, 49, 5)
        yticks = np.linspace(0, 49, 5)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([f'{v:.1f}' for v in np.linspace(1.1, 5.0, 5)])
        ax.set_yticklabels([f'{v:.0%}' for v in np.linspace(0.1, 0.9, 5)])

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('PAF (%)', fontsize=12)

        ax.legend(loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

        return fig

    def generate_table_1(self) -> str:
        """Generate Table 1 for manuscript: PAF by population."""
        lines = [
            "Table 1. Population Attributable Fraction for Algorithmic Discrimination",
            "=" * 80,
            "",
            f"{'Population':<25} | {'P(Exposed)':<12} | {'RR':<6} | {'PAF':<8} | {'N (millions)':<12}",
            "-" * 80,
        ]

        paf_values = self.calculate_all_paf()

        for key, paf in sorted(paf_values.items(), key=lambda x: x[1], reverse=True):
            pop = self.populations[key]
            lines.append(
                f"{pop.name:<25} | {pop.prevalence_exposed:<12.2f} | "
                f"{pop.relative_risk:<6.1f} | {paf*100:<7.1f}% | "
                f"{pop.population_size/1_000_000:<12.1f}"
            )

        lines.extend([
            "-" * 80,
            "",
            "PAF = Population Attributable Fraction",
            "P(Exposed) = Prevalence of exposure to algorithmic screening",
            "RR = Relative risk of adverse outcome given exposure",
            "N = Estimated US population in group",
            "",
            "Interpretation: PAF represents the proportion of adverse outcomes",
            "that would be prevented if algorithmic bias were eliminated.",
        ])

        return "\n".join(lines)


def main():
    """Run PAF analysis and generate outputs."""
    print("=" * 80)
    print("POPULATION ATTRIBUTABLE FRACTION ANALYSIS")
    print("Algorithmic Bias Epidemiology")
    print("=" * 80)

    analysis = PopulationAttributableFractionAnalysis()

    # Print Table 1
    print("\n" + analysis.generate_table_1())

    # Calculate national burden
    print("\n" + "=" * 80)
    print("NATIONAL BURDEN ESTIMATE")
    print("=" * 80)

    burden = analysis.calculate_national_burden(baseline_rate=0.20)

    print(f"\nTotal adverse outcomes attributable to algorithmic bias:")
    print(f"  {burden['total_attributable_cases']:,} cases annually")
    print(f"\nMethodology: {burden['methodology']}")

    print("\nBreakdown by population:")
    for key, data in burden['by_population'].items():
        if key != 'general':
            print(f"  {data['population']}: {data['attributable_to_algorithms']:,} cases")

    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES...")
    print("=" * 80)

    analysis.plot_paf_comparison('paf_comparison.png')
    analysis.plot_sensitivity_heatmap('justice_involved', 'paf_sensitivity_justice.png')
    analysis.plot_sensitivity_heatmap('pwid', 'paf_sensitivity_pwid.png')

    print("\nAnalysis complete.")

    # Export data
    results = {
        'paf_by_population': {k: {'paf': v, **vars(analysis.populations[k])}
                             for k, v in analysis.calculate_all_paf().items()},
        'national_burden': burden,
    }

    # Convert dataclass to dict for JSON serialization
    for k in results['paf_by_population']:
        results['paf_by_population'][k] = {
            key: val for key, val in results['paf_by_population'][k].items()
            if not key.startswith('_')
        }

    with open('paf_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("Results exported to paf_analysis_results.json")


if __name__ == "__main__":
    main()
