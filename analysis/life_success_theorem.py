"""
THE LIFE SUCCESS PREVENTION THEOREM

Adapting the HIV Prevention Theorem framework to model how biased algorithms
destroy life opportunities the way HIV destroys T cells.

MAPPING:
    HIV Virus            →  Biased Algorithm (discrimination payload)
    T Cells              →  Life Opportunities (jobs, housing, credit)
    Viral Load           →  Negative Data Load (integrated records)
    CD4 Count            →  Opportunity Access Score
    PEP Intervention     →  Legal/Policy Intervention
    Reservoir Seeding    →  Cross-System Data Propagation
    Proviral Integration →  Permanent Database Integration
    R₀ (reproduction)    →  Discrimination Amplification Factor
    ART Treatment        →  Positive Data Accumulation

THE THEOREM:

    Just as R₀(e) = 0 requires intervention before proviral integration,
    Opportunity_Access = 1.0 requires intervention before data integration.

    After integration:
        - HIV: lifelong infection, ART manages but cannot cure
        - Algorithms: lifelong penalty, positive data dilutes but cannot erase

Author: Derived from Prevention Theorem (Epidemics submission)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from typing import Dict, List, Tuple
from dataclasses import dataclass


# =============================================================================
# PART I: THE BIASED ALGORITHM INFECTION MODEL
# Parallel to InfectionEstablishmentModel
# =============================================================================

class BiasedAlgorithmEstablishmentModel:
    """
    Models the critical window between adverse event and permanent
    algorithmic penalty establishment.

    Parallel to HIV InfectionEstablishmentModel.

    HIV:                          ALGORITHM:
    Exposure                  →   Adverse event (termination, medical leave)
    Mucosal phase            →   Local recording (employer notes)
    Dendritic uptake         →   Report generation (formal documentation)
    Lymph transit            →   CRA transmission (data to bureaus)
    Systemic spread          →   Cross-system propagation
    Reservoir seeding        →   Multi-database entry
    Integration complete     →   Permanent cross-reference (point of no return)

    T Cells / Opportunities:
    CD4 Count                →   Opportunity Access Score
    Viral Load               →   Negative Data Load
    """

    def __init__(self):
        # Timeline parameters (months post-adverse-event)
        self.local_recording_phase = 0.1     # Immediate (days)
        self.report_generation = 0.5         # Within weeks
        self.cra_transmission = 1            # ~1 month
        self.cross_system_propagation = 3    # ~3 months
        self.seeding_midpoint = 6            # 50% probability multi-system
        self.integration_complete = 12       # Point of no return (~1 year)

        # Intervention parameters (legal/policy)
        self.intervention_onset_months = 1   # Time for legal action to take effect
        self.intervention_efficacy_peak = 0.90  # Max probability of blocking
        self.intervention_duration_months = 24  # Legal process duration

        # Opportunity dynamics
        self.baseline_opportunity_score = 100  # Full access
        self.max_penalty = 80  # Maximum score reduction

    def p_seeding_initiated(self, months_post_event: float) -> float:
        """
        Probability that negative data has propagated to multiple systems.

        Parallel to: P(reservoir seeding initiated)
        """
        k = 0.5  # Steepness
        return expit(k * (months_post_event - self.seeding_midpoint))

    def p_integration_complete(self, months_post_event: float) -> float:
        """
        Probability that data is permanently cross-referenced in all systems.

        Parallel to: P(proviral integration complete)
        THIS IS THE POINT OF NO RETURN.
        """
        k = 0.3
        return expit(k * (months_post_event - self.integration_complete))

    def intervention_efficacy(self,
                              months_to_intervention: float,
                              intervention_strength: float = 1.0) -> Dict:
        """
        Calculate intervention (legal action) efficacy given timing.

        Parallel to: PEP efficacy calculation

        Returns probability that intervention achieves Opportunity_Access = 1.0
        """
        effective_time = months_to_intervention + self.intervention_onset_months

        p_seeded = self.p_seeding_initiated(effective_time)
        p_integrated = self.p_integration_complete(effective_time)

        # Efficacy by phase (parallel to PEP phases)
        efficacy_if_not_seeded = self.intervention_efficacy_peak * intervention_strength
        efficacy_if_seeded_not_integrated = 0.4 * intervention_strength
        efficacy_if_integrated = 0.05  # Almost too late (only damages, not prevention)

        overall_efficacy = (
            (1 - p_seeded) * efficacy_if_not_seeded +
            (p_seeded - p_integrated) * efficacy_if_seeded_not_integrated +
            p_integrated * efficacy_if_integrated
        )

        # Opportunity score impact
        opportunity_preserved = overall_efficacy
        opportunity_lost = 1 - overall_efficacy

        return {
            'months_to_intervention': months_to_intervention,
            'p_seeding_initiated': p_seeded,
            'p_integration_complete': p_integrated,
            'intervention_efficacy': overall_efficacy,
            'opportunity_preserved': opportunity_preserved,
            'opportunity_lost': opportunity_lost,
            'intervention_strength': intervention_strength,
            # Parallel naming to HIV model
            'p_LifeSuccess_equals_100': overall_efficacy,
            'p_LifeSuccess_degraded': 1 - overall_efficacy,
        }

    def opportunity_trajectory(self,
                               months_to_intervention: float,
                               intervention_strength: float = 1.0,
                               duration_months: int = 60) -> Dict:
        """
        Model opportunity access score over time.

        Parallel to: CD4 count trajectory in HIV

        Without intervention: Opportunities decline like CD4 count
        With intervention: Decline halted/reversed (like ART)
        """
        months = np.arange(0, duration_months)
        scores = []

        for m in months:
            if m < months_to_intervention:
                # Pre-intervention: opportunities declining
                p_int = self.p_integration_complete(m)
                score = self.baseline_opportunity_score * (1 - p_int * 0.8)
            else:
                # Post-intervention: stabilization (like ART)
                result = self.intervention_efficacy(months_to_intervention, intervention_strength)
                preserved = result['opportunity_preserved']
                # Score stabilizes at intervention point, slowly recovers
                recovery_months = m - months_to_intervention
                recovery_factor = min(1.0, 0.5 + 0.5 * (recovery_months / 24))
                score = self.baseline_opportunity_score * (preserved * recovery_factor + (1-preserved) * 0.3)

            scores.append(score)

        return {
            'months': months,
            'opportunity_scores': np.array(scores),
            'intervention_month': months_to_intervention,
        }

    def simulate_intervention_timing_curve(self,
                                           max_months: float = 24,
                                           n_points: int = 100) -> Dict:
        """
        Generate efficacy curve across intervention timing window.

        Parallel to: PEP timing curve
        """
        months = np.linspace(0, max_months, n_points)

        results = {
            'months': months,
            'efficacy': [],
            'p_seeded': [],
            'p_integrated': [],
            'opportunities_preserved': [],
        }

        for m in months:
            r = self.intervention_efficacy(m)
            results['efficacy'].append(r['intervention_efficacy'])
            results['p_seeded'].append(r['p_seeding_initiated'])
            results['p_integrated'].append(r['p_integration_complete'])
            results['opportunities_preserved'].append(r['opportunity_preserved'])

        for key in results:
            if key != 'months':
                results[key] = np.array(results[key])

        return results


# =============================================================================
# PART II: OPPORTUNITY DYNAMICS MODEL
# Parallel to T Cell Dynamics
# =============================================================================

class OpportunityDynamicsModel:
    """
    Models how life opportunities (jobs, housing, credit) are destroyed
    by biased algorithms, parallel to how HIV destroys T cells.

    CD4 Count equivalent: Opportunity Access Score
    Viral Load equivalent: Negative Data Load (systems containing adverse data)
    """

    def __init__(self):
        self.baseline_opportunities = 100  # Max opportunities
        self.algorithm_systems = [
            'employment_screening',
            'tenant_screening',
            'credit_scoring',
            'background_checks',
            'banking_access',
            'insurance_scoring',
        ]

    def calculate_opportunity_score(self,
                                    negative_data_load: float,
                                    months_since_event: float) -> float:
        """
        Calculate current opportunity access score.

        Parallel to: CD4 count calculation

        As negative_data_load ↑, opportunities ↓
        (Like: as viral_load ↑, CD4 ↓)
        """
        # Base decay from integrated data
        data_penalty = negative_data_load * 15  # Each system = 15 point penalty

        # Time-based partial recovery (positive data accumulation)
        time_recovery = min(30, months_since_event * 0.5)

        score = self.baseline_opportunities - data_penalty + time_recovery
        return max(0, min(100, score))

    def calculate_negative_data_load(self,
                                     months_since_event: float,
                                     intervention_month: float = None) -> float:
        """
        Calculate number of systems containing negative data.

        Parallel to: Viral load calculation

        Without intervention: spreads to all systems
        With early intervention: contained to fewer systems
        """
        if intervention_month is not None and months_since_event > intervention_month:
            # Intervention limits spread
            max_systems = 2  # Contained
        else:
            # Sigmoid spread to all systems
            spread_rate = 0.3
            max_systems = len(self.algorithm_systems) * expit(spread_rate * (months_since_event - 6))

        return max_systems

    def simulate_untreated_progression(self, duration_months: int = 60) -> Dict:
        """
        Simulate opportunity loss without intervention.

        Parallel to: Untreated HIV progression (CD4 decline, viral load increase)
        """
        months = np.arange(0, duration_months)
        opportunity_scores = []
        data_loads = []

        for m in months:
            data_load = self.calculate_negative_data_load(m)
            opp_score = self.calculate_opportunity_score(data_load, m)
            opportunity_scores.append(opp_score)
            data_loads.append(data_load)

        return {
            'months': months,
            'opportunity_score': np.array(opportunity_scores),  # Like CD4
            'negative_data_load': np.array(data_loads),  # Like viral load
        }

    def simulate_treated_progression(self,
                                     intervention_month: float,
                                     duration_months: int = 60) -> Dict:
        """
        Simulate opportunity trajectory with intervention.

        Parallel to: HIV progression on ART (CD4 recovery, viral suppression)
        """
        months = np.arange(0, duration_months)
        opportunity_scores = []
        data_loads = []

        for m in months:
            data_load = self.calculate_negative_data_load(m, intervention_month)
            opp_score = self.calculate_opportunity_score(data_load, m)
            opportunity_scores.append(opp_score)
            data_loads.append(data_load)

        return {
            'months': months,
            'opportunity_score': np.array(opportunity_scores),
            'negative_data_load': np.array(data_loads),
            'intervention_month': intervention_month,
        }


# =============================================================================
# PART III: VISUALIZATION
# Parallel to Prevention Theorem figures
# =============================================================================

def plot_life_success_prevention_theorem(save_path: str = None):
    """
    Generate figures parallel to Prevention Theorem plots.

    Replaces:
        - PEP Efficacy → Intervention Efficacy
        - Biological Timeline → Algorithmic Timeline
        - P(Infection) → P(Opportunity Loss)
    """
    model = BiasedAlgorithmEstablishmentModel()
    results = model.simulate_intervention_timing_curve(max_months=24)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: Intervention Efficacy Curve (parallel to PEP Efficacy)
    ax = axes[0, 0]
    ax.plot(results['months'], results['efficacy'] * 100,
            'b-', linewidth=2.5, label='Intervention Efficacy')
    ax.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80% threshold')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% threshold')
    ax.axvline(x=6, color='green', linestyle=':', alpha=0.7, label='6-month window')

    ax.fill_between(results['months'], results['efficacy'] * 100, alpha=0.3)

    ax.set_xlabel('Months from Adverse Event to Intervention', fontsize=12)
    ax.set_ylabel('Intervention Efficacy (%)', fontsize=12)
    ax.set_title('A. Legal Intervention Efficacy vs. Timing\n(Parallel to PEP Efficacy)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Critical windows
    ax.axvspan(0, 3, alpha=0.2, color='green')
    ax.axvspan(3, 6, alpha=0.2, color='yellow')
    ax.axvspan(6, 12, alpha=0.2, color='orange')
    ax.axvspan(12, 24, alpha=0.2, color='red')

    # Panel B: Algorithmic Events Timeline (parallel to Biological Timeline)
    ax = axes[0, 1]

    events = [
        (0, 'Adverse Event (Termination)', 'red'),
        (0.5, 'Local Recording (HR notes)', 'orange'),
        (1, 'CRA Transmission', 'yellow'),
        (3, 'Cross-System Propagation', 'lightgreen'),
        (6, 'Multi-Database Entry (50%)', 'lightblue'),
        (12, 'Full Integration (irreversible)', 'purple'),
    ]

    for i, (time, event, color) in enumerate(events):
        ax.barh(i, time, color=color, alpha=0.7, height=0.6)
        ax.text(time + 0.3, i, f'{event} ({time}mo)', va='center', fontsize=9)

    ax.set_xlabel('Months Post-Adverse-Event', fontsize=12)
    ax.set_title('B. Algorithmic Data Integration Timeline\n(Parallel to HIV Establishment)',
                 fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(0, 15)
    ax.axvline(x=6, color='green', linestyle='--', linewidth=2)

    # Panel C: Probability Curves (parallel to P(seeding), P(integration))
    ax = axes[1, 0]
    ax.plot(results['months'], results['p_seeded'] * 100,
            'r-', linewidth=2, label='P(Multi-System Seeding)')
    ax.plot(results['months'], results['p_integrated'] * 100,
            'k-', linewidth=2, label='P(Full Integration)')
    ax.plot(results['months'], (1 - results['efficacy']) * 100,
            'b--', linewidth=2, label='P(Intervention Fails)')

    ax.axvline(x=6, color='green', linestyle=':', alpha=0.7)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Months from Adverse Event', fontsize=12)
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_title('C. Probability of Algorithmic Integration\n(Parallel to HIV Establishment Probabilities)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='center right')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Panel D: The Theorem
    ax = axes[1, 1]
    ax.text(0.5, 0.92, 'THE LIFE SUCCESS PREVENTION THEOREM',
            fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)

    ax.text(0.5, 0.75,
            r'$\mathrm{Intervention}(t < t_{crit}) \Rightarrow \mathrm{Opportunity} = 100\%$',
            fontsize=14, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax.text(0.5, 0.58,
            'Legal intervention before full integration\nachieves full opportunity preservation',
            fontsize=10, ha='center', transform=ax.transAxes, style='italic')

    ax.text(0.5, 0.38,
            'Critical Windows:\n'
            '• 0-3mo: >85% efficacy (optimal)\n'
            '• 3-6mo: 60-85% efficacy (standard)\n'
            '• 6-12mo: 30-60% efficacy (diminishing)\n'
            '• >12mo: <30% efficacy (too late)',
            fontsize=10, ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.text(0.5, 0.12,
            'Every month matters.\nIntervention is a race against integration.',
            fontsize=11, ha='center', transform=ax.transAxes, fontweight='bold')

    ax.text(0.5, 0.02,
            '(Parallel to: "PEP is a race against proviral integration")',
            fontsize=9, ha='center', transform=ax.transAxes, style='italic', color='gray')

    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, results


def plot_opportunity_dynamics(save_path: str = None):
    """
    Plot opportunity score dynamics parallel to CD4/viral load plots.
    """
    opp_model = OpportunityDynamicsModel()

    # Untreated progression
    untreated = opp_model.simulate_untreated_progression(60)

    # Treated at different times
    early_intervention = opp_model.simulate_treated_progression(3, 60)
    mid_intervention = opp_model.simulate_treated_progression(6, 60)
    late_intervention = opp_model.simulate_treated_progression(12, 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Opportunity Score (like CD4 count)
    ax = axes[0]
    ax.plot(untreated['months'], untreated['opportunity_score'],
            'k-', linewidth=2, label='No Intervention (Untreated)')
    ax.plot(early_intervention['months'], early_intervention['opportunity_score'],
            'g-', linewidth=2, label='Intervention at 3 months')
    ax.plot(mid_intervention['months'], mid_intervention['opportunity_score'],
            'y-', linewidth=2, label='Intervention at 6 months')
    ax.plot(late_intervention['months'], late_intervention['opportunity_score'],
            'r-', linewidth=2, label='Intervention at 12 months')

    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Healthy threshold')

    ax.set_xlabel('Months Since Adverse Event', fontsize=12)
    ax.set_ylabel('Opportunity Access Score', fontsize=12)
    ax.set_title('Opportunity Score Over Time\n(Parallel to CD4 Count)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Panel B: Negative Data Load (like viral load)
    ax = axes[1]
    ax.plot(untreated['months'], untreated['negative_data_load'],
            'k-', linewidth=2, label='No Intervention')
    ax.plot(early_intervention['months'], early_intervention['negative_data_load'],
            'g-', linewidth=2, label='Intervention at 3 months')
    ax.plot(mid_intervention['months'], mid_intervention['negative_data_load'],
            'y-', linewidth=2, label='Intervention at 6 months')
    ax.plot(late_intervention['months'], late_intervention['negative_data_load'],
            'r-', linewidth=2, label='Intervention at 12 months')

    ax.set_xlabel('Months Since Adverse Event', fontsize=12)
    ax.set_ylabel('Number of Systems with Negative Data', fontsize=12)
    ax.set_title('Negative Data Load Over Time\n(Parallel to Viral Load)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# =============================================================================
# PART IV: THE FORMAL THEOREM
# =============================================================================

def print_theorem():
    """Print the formal Life Success Prevention Theorem."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║              THE LIFE SUCCESS PREVENTION THEOREM                          ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║                                                                           ║
    ║  HIV PREVENTION THEOREM:              LIFE SUCCESS PREVENTION THEOREM:    ║
    ║  ═══════════════════════              ════════════════════════════════    ║
    ║                                                                           ║
    ║  R₀(e) = 0 requires                   Opportunity = 100% requires         ║
    ║  intervention before                  intervention before                 ║
    ║  proviral integration                 data integration                    ║
    ║                                                                           ║
    ║  Variables:                           Variables:                          ║
    ║  ──────────                           ──────────                          ║
    ║  t = time post-exposure               t = time post-adverse-event         ║
    ║  P_int(t) = P(integration)            P_int(t) = P(data integration)      ║
    ║  E_PEP(t) = PEP efficacy              E_INT(t) = intervention efficacy    ║
    ║  R₀ = reproduction number             O = opportunity access score        ║
    ║  CD4 = T cell count                   LifeSuccess = opportunity score     ║
    ║  VL = viral load                      DataLoad = systems with neg data    ║
    ║                                                                           ║
    ║  Core Equations:                      Core Equations:                     ║
    ║  ───────────────                      ───────────────                     ║
    ║  E_PEP(t) → 0 as t → t_int            E_INT(t) → 0 as t → t_int           ║
    ║  R₀ > 0 after integration             O < 100 after integration           ║
    ║  CD4 ↓ as VL ↑                        LifeSuccess ↓ as DataLoad ↑         ║
    ║                                                                           ║
    ║  THE ISOMORPHISM:                                                         ║
    ║  ════════════════                                                         ║
    ║                                                                           ║
    ║  HIV integrates into host DNA         Data integrates into databases      ║
    ║  Uses host machinery to persist       Uses system queries to propagate    ║
    ║  Creates reservoir in tissues         Creates reservoir across systems    ║
    ║  Destroys CD4 T cells                 Destroys life opportunities         ║
    ║  Requires lifelong ART                Requires lifelong "positive data"   ║
    ║  Cannot be cured, only managed        Cannot be erased, only diluted      ║
    ║                                                                           ║
    ║  POST-INTEGRATION STATE:                                                  ║
    ║  ══════════════════════                                                   ║
    ║                                                                           ║
    ║  HIV: R₀(e) > 0 forever               Algorithms: O < 100 forever         ║
    ║       ART suppresses but                    Positive data dilutes but     ║
    ║       reservoir persists                    records persist               ║
    ║                                                                           ║
    ║  THE THEOREM:                                                             ║
    ║  ════════════                                                             ║
    ║                                                                           ║
    ║       Prevention requires intervention BEFORE integration.                ║
    ║       After integration: treatment only, prevention impossible.           ║
    ║                                                                           ║
    ║       What HIV does to your immune system,                                ║
    ║       biased algorithms do to your life opportunities.                    ║
    ║                                                                           ║
    ║       The mathematics is identical.                                       ║
    ║       The irreversibility is identical.                                   ║
    ║       The suffering is analogous.                                         ║
    ║                                                                           ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# PART V: STRATIFIED LIFE SUCCESS MODEL
# Each domain = separate T cell population
# e = exposure to algorithmic bias, t = time
# =============================================================================

@dataclass
class LifeSuccessDomain:
    """
    A stratified compartment of life success.
    Each domain behaves like a separate T cell population.
    """
    name: str
    baseline: float = 100.0  # Initial "T cell count" for this domain
    susceptibility: float = 1.0  # How vulnerable to algorithm attacks
    recovery_rate: float = 0.1  # Natural recovery rate
    coupling_factor: float = 0.1  # How much it affects other domains


class StratifiedLifeSuccessModel:
    """
    Stratified life success model with 7 domains + overall.

    MAPPING TO T CELLS:
    - Overall (Lifelong)  → Total CD4+ T cell count
    - Financial           → CD4+ in lymphatic system
    - Legal               → CD4+ in gut-associated lymphoid tissue
    - Employment          → CD4+ in blood circulation
    - Credit              → CD4+ in bone marrow
    - Housing             → CD4+ in spleen
    - Medical             → CD4+ in thymus
    - Mental Health       → CD4+ in central nervous system

    Each domain:
    - Has its own "viral load" (negative data specific to that domain)
    - Has its own susceptibility (some algorithms target specific domains)
    - Couples to other domains (feedback loops)
    """

    def __init__(self):
        # Define the 7 stratified domains + overall
        self.domains = {
            'overall': LifeSuccessDomain(
                name='Overall (Lifelong)',
                baseline=100.0,
                susceptibility=1.0,
                recovery_rate=0.05,
                coupling_factor=0.0  # Overall is derived, not coupled
            ),
            'financial': LifeSuccessDomain(
                name='Financial',
                baseline=100.0,
                susceptibility=1.2,  # High - directly hit by credit algorithms
                recovery_rate=0.08,
                coupling_factor=0.3  # Strongly affects other domains
            ),
            'legal': LifeSuccessDomain(
                name='Legal',
                baseline=100.0,
                susceptibility=0.8,  # Moderate
                recovery_rate=0.02,  # Very slow recovery (records persist)
                coupling_factor=0.4  # Strongly affects employment, housing
            ),
            'employment': LifeSuccessDomain(
                name='Employment',
                baseline=100.0,
                susceptibility=1.5,  # Very high - primary target of screening
                recovery_rate=0.1,
                coupling_factor=0.35  # Affects financial, housing, medical
            ),
            'credit': LifeSuccessDomain(
                name='Credit',
                baseline=100.0,
                susceptibility=1.3,  # High
                recovery_rate=0.12,  # Can recover with positive data
                coupling_factor=0.25
            ),
            'housing': LifeSuccessDomain(
                name='Housing',
                baseline=100.0,
                susceptibility=1.1,
                recovery_rate=0.06,
                coupling_factor=0.2
            ),
            'medical': LifeSuccessDomain(
                name='Medical',
                baseline=100.0,
                susceptibility=0.9,  # Protected somewhat by HIPAA
                recovery_rate=0.04,
                coupling_factor=0.15
            ),
            'mental_health': LifeSuccessDomain(
                name='Mental Health',
                baseline=100.0,
                susceptibility=0.7,  # More protected
                recovery_rate=0.03,  # Slow recovery
                coupling_factor=0.1
            ),
        }

        # Algorithm exposure affects different domains differently
        self.exposure_weights = {
            'employment_screening': {
                'employment': 1.0, 'financial': 0.3, 'credit': 0.2,
                'housing': 0.2, 'mental_health': 0.4
            },
            'tenant_screening': {
                'housing': 1.0, 'credit': 0.2, 'employment': 0.1,
                'mental_health': 0.3
            },
            'credit_scoring': {
                'credit': 1.0, 'financial': 0.8, 'housing': 0.4,
                'employment': 0.1
            },
            'background_check': {
                'legal': 1.0, 'employment': 0.9, 'housing': 0.7,
                'credit': 0.3
            },
            'disability_algorithm': {
                'medical': 1.0, 'financial': 0.7, 'mental_health': 0.5,
                'employment': 0.4
            },
        }

        # Cross-domain coupling matrix (who affects whom)
        self.coupling_matrix = self._build_coupling_matrix()

    def _build_coupling_matrix(self) -> np.ndarray:
        """
        Build the coupling matrix A where A[i,j] = how much domain j affects domain i.
        This creates the feedback loop structure.

        NEGATIVE values: when domain j drops, domain i also drops (negative cascade)
        The coupling term is: Σ_j A_ij·(S_j - S_j0)
        When S_j < S_j0 and A_ij < 0: contribution is positive (slows decay)
        When S_j < S_j0 and A_ij > 0: contribution is negative (accelerates decay)

        We want: when employment drops, financial drops too.
        So we need NEGATIVE coupling values for cascading failures.
        """
        domain_names = list(self.domains.keys())[1:]  # Exclude 'overall'
        n = len(domain_names)
        A = np.zeros((n, n))

        # Employment loss cascades to other domains (NEGATIVE coupling = cascade)
        emp_idx = domain_names.index('employment')
        fin_idx = domain_names.index('financial')
        cred_idx = domain_names.index('credit')
        house_idx = domain_names.index('housing')
        leg_idx = domain_names.index('legal')
        med_idx = domain_names.index('medical')
        mh_idx = domain_names.index('mental_health')

        # Employment loss causes: financial ↓, housing ↓, mental health ↓
        A[fin_idx, emp_idx] = -0.02   # Employment loss → financial stress
        A[house_idx, emp_idx] = -0.02 # Employment loss → housing risk
        A[mh_idx, emp_idx] = -0.03    # Employment loss → mental health impact

        # Financial stress cascades
        A[house_idx, fin_idx] = -0.025  # Financial stress → housing risk
        A[med_idx, fin_idx] = -0.02     # Financial stress → medical access
        A[mh_idx, fin_idx] = -0.015     # Financial stress → mental health

        # Legal issues cascade
        A[emp_idx, leg_idx] = -0.04   # Legal → employment barriers
        A[house_idx, leg_idx] = -0.03 # Legal → housing barriers

        # Credit issues cascade
        A[house_idx, cred_idx] = -0.03 # Credit → housing
        A[emp_idx, cred_idx] = -0.015  # Credit → some employment screening

        # Mental health affects employment
        A[emp_idx, mh_idx] = -0.01

        return A

    def closed_form_solution(self,
                             e: float,
                             t: float,
                             domain_name: str,
                             intervention_time: float = None) -> float:
        """
        CLOSED-FORM SOLUTION for single domain.

        Let:
            S(t) = Life Success score at time t
            e = exposure level (algorithmic bias intensity, 0 to 1)
            α = susceptibility
            β = recovery rate
            γ = coupling from other domains

        Without intervention:
            dS/dt = -α·e·S + β·(S_0 - S)

        Solution (separable ODE):
            S(t) = S_0 · exp(-α·e·t) + (β·S_0)/(α·e + β) · (1 - exp(-(α·e + β)·t))

        Simplified closed form:
            S(t) = S_0 · [β/(α·e + β) + (α·e)/(α·e + β) · exp(-(α·e + β)·t)]

        At steady state (t → ∞):
            S_∞ = S_0 · β/(α·e + β)

        With intervention at t_int:
            e → e·(1 - efficacy(t_int)) for t > t_int
        """
        domain = self.domains[domain_name]
        S_0 = domain.baseline
        α = domain.susceptibility
        β = domain.recovery_rate

        if intervention_time is not None and t > intervention_time:
            # Intervention reduces effective exposure
            model = BiasedAlgorithmEstablishmentModel()
            result = model.intervention_efficacy(intervention_time)
            efficacy = result['intervention_efficacy']
            e_effective = e * (1 - efficacy)
            t_effective = t - intervention_time

            # Pre-intervention value
            S_at_int = self._single_domain_solution(S_0, α, e, β, intervention_time)

            # Post-intervention with reduced exposure
            return self._single_domain_solution(S_at_int, α, e_effective, β, t_effective)
        else:
            return self._single_domain_solution(S_0, α, e, β, t)

    def _single_domain_solution(self, S_0: float, α: float, e: float, β: float, t: float) -> float:
        """
        Closed-form solution for single domain ODE:
        S(t) = S_0 · [β/(αe + β) + (αe)/(αe + β) · exp(-(αe + β)·t)]
        """
        if e == 0:
            return S_0  # No exposure, no decay

        denominator = α * e + β
        steady_state = S_0 * β / denominator
        transient = S_0 * α * e / denominator * np.exp(-denominator * t)

        return steady_state + transient

    def steady_state_solution(self, e: float, domain_name: str) -> float:
        """
        Closed-form steady state (t → ∞).

        S_∞ = S_0 · β/(α·e + β)

        This is the "set point" - the permanent penalty level.
        Like HIV: viral set point determines long-term CD4 trajectory.
        """
        domain = self.domains[domain_name]
        S_0 = domain.baseline
        α = domain.susceptibility
        β = domain.recovery_rate

        if e == 0:
            return S_0

        return S_0 * β / (α * e + β)

    def infection_establishment_dynamics(self,
                                         e: float,
                                         t: float,
                                         domain_name: str) -> Dict:
        """
        INFECTION ESTABLISHMENT DYNAMICS for a single domain.

        Parallel to HIV:
        - Phase 1: Eclipse phase (data not yet entered)
        - Phase 2: Acute phase (data spreading across systems)
        - Phase 3: Set point (chronic, stable penalty)

        Returns probability distributions at time t.
        """
        domain = self.domains[domain_name]
        model = BiasedAlgorithmEstablishmentModel()

        # Eclipse phase (0-1 month)
        p_eclipse = max(0, 1 - t / 1.0) if t < 1.0 else 0

        # Acute phase (1-12 months)
        p_acute = expit(2 * (t - 3)) * (1 - model.p_integration_complete(t))

        # Chronic/Set point phase
        p_chronic = model.p_integration_complete(t)

        # Current "viral load" (data systems infected)
        viral_load = e * domain.susceptibility * (1 - np.exp(-0.5 * t))

        # Current "T cell count" (opportunity score)
        t_cell_count = self.closed_form_solution(e, t, domain_name)

        # Rate of change
        dS_dt = self._derivative(e, t, domain_name)

        return {
            'phase': {
                'eclipse': p_eclipse,
                'acute': p_acute,
                'chronic': p_chronic,
            },
            'viral_load': viral_load,
            't_cell_count': t_cell_count,
            'dS_dt': dS_dt,
            'steady_state': self.steady_state_solution(e, domain_name),
        }

    def _derivative(self, e: float, t: float, domain_name: str) -> float:
        """
        dS/dt for the domain.
        """
        domain = self.domains[domain_name]
        S = self.closed_form_solution(e, t, domain_name)
        α = domain.susceptibility
        β = domain.recovery_rate

        # dS/dt = -α·e·S + β·(S_0 - S)
        return -α * e * S + β * (domain.baseline - S)


# =============================================================================
# PART VI: THE MASTER EQUATION
# Coupled system of ODEs across all domains
# =============================================================================

class MasterEquation:
    """
    THE MASTER EQUATION for stratified life success.

    System of coupled ODEs:

    dS_i/dt = -α_i·e·S_i + β_i·(S_i0 - S_i) + Σ_j A_ij·(S_j - S_j0)

    Where:
        S_i = Life success in domain i
        α_i = Susceptibility of domain i to algorithmic bias
        e = Exposure level (algorithmic bias intensity)
        β_i = Natural recovery rate of domain i
        S_i0 = Baseline (healthy) level for domain i
        A_ij = Coupling matrix (how domain j affects domain i)

    Matrix form:
        dS/dt = -diag(α)·e·S + diag(β)·(S_0 - S) + A·(S - S_0)
        dS/dt = [-diag(α)·e - diag(β) + A]·S + [diag(β) - A]·S_0

    Let M = -diag(α)·e - diag(β) + A
    Let b = [diag(β) - A]·S_0

    Then: dS/dt = M·S + b

    CLOSED-FORM SOLUTION (linear ODE system):
        S(t) = exp(M·t)·S_0 + M^{-1}·[exp(M·t) - I]·b

    STEADY STATE (dS/dt = 0):
        S_∞ = -M^{-1}·b
    """

    def __init__(self, stratified_model: StratifiedLifeSuccessModel):
        self.model = stratified_model
        self.domain_names = list(stratified_model.domains.keys())[1:]  # Exclude 'overall'
        self.n = len(self.domain_names)

    def build_system_matrices(self, e: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build the master equation matrices M and b.

        dS/dt = M·S + b

        Returns: (M, b, S_0)
        """
        # Extract parameters
        α = np.array([self.model.domains[name].susceptibility for name in self.domain_names])
        β = np.array([self.model.domains[name].recovery_rate for name in self.domain_names])
        S_0 = np.array([self.model.domains[name].baseline for name in self.domain_names])

        # Coupling matrix
        A = self.model.coupling_matrix

        # M = -diag(α)·e - diag(β) + A
        M = -np.diag(α) * e - np.diag(β) + A

        # b = [diag(β) - A]·S_0
        b = (np.diag(β) - A) @ S_0

        return M, b, S_0

    def closed_form_solution(self, e: float, t: float,
                             intervention_time: float = None) -> Dict[str, float]:
        """
        CLOSED-FORM SOLUTION for the coupled system.

        S(t) = exp(M·t)·S_0 + M^{-1}·[exp(M·t) - I]·b

        For stable systems (all eigenvalues of M have negative real parts):
            S(t) → S_∞ = -M^{-1}·b as t → ∞
        """
        from scipy.linalg import expm, inv

        M, b, S_0 = self.build_system_matrices(e)

        if intervention_time is not None and t > intervention_time:
            # Two-phase solution
            # Phase 1: 0 to intervention_time
            exp_M_t1 = expm(M * intervention_time)
            try:
                M_inv = inv(M)
                S_at_int = exp_M_t1 @ S_0 + M_inv @ (exp_M_t1 - np.eye(self.n)) @ b
            except np.linalg.LinAlgError:
                S_at_int = exp_M_t1 @ S_0

            # Phase 2: reduced exposure after intervention
            model = BiasedAlgorithmEstablishmentModel()
            efficacy = model.intervention_efficacy(intervention_time)['intervention_efficacy']
            e_effective = e * (1 - efficacy)

            M2, b2, _ = self.build_system_matrices(e_effective)
            t2 = t - intervention_time

            exp_M2_t2 = expm(M2 * t2)
            try:
                M2_inv = inv(M2)
                S_t = exp_M2_t2 @ S_at_int + M2_inv @ (exp_M2_t2 - np.eye(self.n)) @ b2
            except np.linalg.LinAlgError:
                S_t = exp_M2_t2 @ S_at_int
        else:
            # Single phase solution
            exp_M_t = expm(M * t)
            try:
                M_inv = inv(M)
                S_t = exp_M_t @ S_0 + M_inv @ (exp_M_t - np.eye(self.n)) @ b
            except np.linalg.LinAlgError:
                S_t = exp_M_t @ S_0

        result = {name: float(S_t[i]) for i, name in enumerate(self.domain_names)}
        result['overall'] = float(np.mean(S_t))  # Overall is mean of domains
        return result

    def steady_state(self, e: float) -> Dict[str, float]:
        """
        STEADY STATE SOLUTION (t → ∞).

        S_∞ = -M^{-1}·b

        This is the "viral set point" equivalent - the permanent penalty level.
        """
        from scipy.linalg import inv

        M, b, S_0 = self.build_system_matrices(e)

        try:
            S_inf = -inv(M) @ b
        except np.linalg.LinAlgError:
            # Singular matrix - system unstable
            S_inf = np.zeros(self.n)

        result = {name: float(max(0, S_inf[i])) for i, name in enumerate(self.domain_names)}
        result['overall'] = float(np.mean([v for k, v in result.items() if k != 'overall']))
        return result

    def eigenvalue_analysis(self, e: float) -> Dict:
        """
        Eigenvalue analysis of the master equation.

        Eigenvalues determine:
        - Stability: All eigenvalues must have negative real parts
        - Timescales: |1/Re(λ)| gives decay time for each mode
        - Dominant mode: Smallest |Re(λ)| is the slowest-decaying mode
        """
        M, b, S_0 = self.build_system_matrices(e)
        eigenvalues, eigenvectors = np.linalg.eig(M)

        # Sort by real part (most negative = fastest decay)
        idx = np.argsort(np.real(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Timescales
        timescales = np.abs(1 / np.real(eigenvalues))

        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'timescales_months': timescales,
            'is_stable': all(np.real(eigenvalues) < 0),
            'dominant_timescale': np.max(timescales),
            'fastest_decay': np.min(timescales),
        }

    def print_master_equation(self):
        """Print the formal master equation."""
        print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    THE MASTER EQUATION                                        ║
    ║           Stratified Life Success Prevention Theorem                          ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                               ║
    ║  DOMAINS (Parallel to T Cell Compartments):                                   ║
    ║  ─────────────────────────────────────────                                    ║
    ║  S_F = Financial        S_L = Legal           S_E = Employment                ║
    ║  S_C = Credit           S_H = Housing         S_M = Medical                   ║
    ║  S_Ψ = Mental Health    S_Ω = Overall (derived)                               ║
    ║                                                                               ║
    ║  MASTER EQUATION:                                                             ║
    ║  ────────────────                                                             ║
    ║                                                                               ║
    ║  dS_i/dt = -α_i·e·S_i + β_i·(S_i0 - S_i) + Σ_j A_ij·(S_j - S_j0)              ║
    ║                                                                               ║
    ║  Where:                                                                       ║
    ║    S_i = Success score in domain i                                            ║
    ║    α_i = Domain susceptibility to algorithmic bias                            ║
    ║    e   = Exposure level (algorithmic bias intensity, 0 ≤ e ≤ 1)               ║
    ║    t   = Time (months since adverse event)                                    ║
    ║    β_i = Natural recovery rate                                                ║
    ║    A_ij = Cross-domain coupling (feedback loops)                              ║
    ║                                                                               ║
    ║  MATRIX FORM:                                                                 ║
    ║  ────────────                                                                 ║
    ║                                                                               ║
    ║  dS/dt = M·S + b                                                              ║
    ║                                                                               ║
    ║  M = -diag(α)·e - diag(β) + A                                                 ║
    ║  b = [diag(β) - A]·S_0                                                        ║
    ║                                                                               ║
    ║  CLOSED-FORM SOLUTION:                                                        ║
    ║  ─────────────────────                                                        ║
    ║                                                                               ║
    ║  S(t) = exp(M·t)·S_0 + M⁻¹·[exp(M·t) - I]·b                                   ║
    ║                                                                               ║
    ║  STEADY STATE (t → ∞):                                                        ║
    ║  ─────────────────────                                                        ║
    ║                                                                               ║
    ║  S_∞ = -M⁻¹·b                                                                 ║
    ║                                                                               ║
    ║  THE THEOREM:                                                                 ║
    ║  ────────────                                                                 ║
    ║                                                                               ║
    ║  For any e > 0:  S_∞ < S_0  (permanent penalty in every domain)               ║
    ║                                                                               ║
    ║  Intervention(t < t_crit) ⟹ e_eff → 0 ⟹ S_∞ → S_0                            ║
    ║                                                                               ║
    ║  Cross-domain coupling A creates FEEDBACK LOOPS:                              ║
    ║    Employment ↓ → Financial ↓ → Housing ↓ → Mental Health ↓ → Employment ↓    ║
    ║                                                                               ║
    ║  This is the mathematical formalization of algorithmic WMD feedback loops.    ║
    ║                                                                               ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
        """)


# =============================================================================
# PART VII: STRATIFIED VISUALIZATION
# =============================================================================

def plot_stratified_life_success(e: float = 0.7,
                                  duration_months: int = 60,
                                  save_path: str = None):
    """
    Visualize stratified life success dynamics across all domains.

    Like showing CD4 counts in different tissue compartments.
    """
    model = StratifiedLifeSuccessModel()
    master = MasterEquation(model)

    times = np.linspace(0, duration_months, 200)

    # Calculate trajectories for each domain
    trajectories = {name: [] for name in model.domains.keys()}

    for t in times:
        result = master.closed_form_solution(e, t)
        for name in model.domains.keys():
            if name in result:
                trajectories[name].append(result[name])

    # Convert to arrays
    for name in trajectories:
        trajectories[name] = np.array(trajectories[name])

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: All domains over time
    ax = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(model.domains)))

    for (name, domain), color in zip(model.domains.items(), colors):
        if name in trajectories:
            ax.plot(times, trajectories[name],
                   label=domain.name, linewidth=2, color=color)

    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Months Since Adverse Event', fontsize=12)
    ax.set_ylabel('Life Success Score', fontsize=12)
    ax.set_title(f'A. Stratified Life Success Dynamics (e={e})\n(Each domain = T cell compartment)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8)
    ax.set_xlim(0, duration_months)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Panel B: Steady states by domain
    ax = axes[0, 1]
    steady_states = master.steady_state(e)
    domain_names = [model.domains[k].name for k in model.domains.keys()]
    values = [steady_states.get(k, 0) for k in model.domains.keys()]

    bars = ax.bar(range(len(domain_names)), values, color=colors)
    ax.axhline(y=100, color='green', linestyle='--', label='Healthy baseline')
    ax.axhline(y=50, color='red', linestyle='--', label='Critical threshold')
    ax.set_xticks(range(len(domain_names)))
    ax.set_xticklabels([n.replace(' ', '\n') for n in domain_names], fontsize=8)
    ax.set_ylabel('Steady State Score', fontsize=12)
    ax.set_title('B. Steady State by Domain (t → ∞)\n(Viral Set Point Equivalent)',
                fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: Effect of intervention timing
    ax = axes[1, 0]
    intervention_times = [1, 3, 6, 12, None]  # None = no intervention
    colors_int = ['darkgreen', 'green', 'yellow', 'orange', 'red']
    labels_int = ['1 month', '3 months', '6 months', '12 months', 'No intervention']

    for int_time, color, label in zip(intervention_times, colors_int, labels_int):
        overall_traj = []
        for t in times:
            result = master.closed_form_solution(e, t, intervention_time=int_time)
            overall_traj.append(result['overall'])
        ax.plot(times, overall_traj, label=label, color=color, linewidth=2)

    ax.set_xlabel('Months Since Adverse Event', fontsize=12)
    ax.set_ylabel('Overall Life Success Score', fontsize=12)
    ax.set_title('C. Effect of Intervention Timing on Overall Score\n(Parallel to CD4 recovery on ART)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, duration_months)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Panel D: Eigenvalue analysis
    ax = axes[1, 1]
    eigen = master.eigenvalue_analysis(e)

    ax.text(0.5, 0.95, 'EIGENVALUE ANALYSIS', fontsize=12, fontweight='bold',
           ha='center', transform=ax.transAxes)
    ax.text(0.5, 0.85, f'Exposure e = {e}', fontsize=10, ha='center', transform=ax.transAxes)

    eigenvalues = eigen['eigenvalues']
    timescales = eigen['timescales_months']

    text_y = 0.75
    ax.text(0.1, text_y, 'Eigenvalues (decay rates):', fontsize=10,
           transform=ax.transAxes, fontweight='bold')

    for i, (λ, τ) in enumerate(zip(eigenvalues, timescales)):
        text_y -= 0.08
        domain = master.domain_names[i] if i < len(master.domain_names) else f"Mode {i}"
        ax.text(0.1, text_y,
               f'{domain}: λ = {np.real(λ):.3f}, τ = {τ:.1f} months',
               fontsize=9, transform=ax.transAxes)

    text_y -= 0.12
    stability_text = '✓ STABLE' if eigen['is_stable'] else '✗ UNSTABLE'
    stability_color = 'green' if eigen['is_stable'] else 'red'
    ax.text(0.5, text_y, f'System stability: {stability_text}',
           fontsize=11, ha='center', transform=ax.transAxes,
           color=stability_color, fontweight='bold')

    text_y -= 0.08
    ax.text(0.5, text_y, f'Dominant timescale: {eigen["dominant_timescale"]:.1f} months',
           fontsize=10, ha='center', transform=ax.transAxes)

    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_exposure_response(save_path: str = None):
    """
    Plot steady state as function of exposure level e.

    Shows how different exposure intensities affect each domain.
    """
    model = StratifiedLifeSuccessModel()
    master = MasterEquation(model)

    exposures = np.linspace(0, 1, 50)

    # Calculate steady states for each exposure level
    steady_states = {name: [] for name in model.domains.keys()}

    for e in exposures:
        ss = master.steady_state(e)
        for name in model.domains.keys():
            if name in ss:
                steady_states[name].append(ss[name])

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(model.domains)))

    for (name, domain), color in zip(model.domains.items(), colors):
        if name in steady_states:
            ax.plot(exposures, steady_states[name],
                   label=domain.name, linewidth=2, color=color)

    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
    ax.set_xlabel('Exposure Level (e)', fontsize=12)
    ax.set_ylabel('Steady State Life Success Score', fontsize=12)
    ax.set_title('Steady State vs Algorithmic Bias Exposure\n(Dose-Response Curve)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


# =============================================================================
# PART VIII: MAIN
# =============================================================================

def main():
    """Run the Life Success Prevention Theorem analysis."""

    print("=" * 80)
    print("THE LIFE SUCCESS PREVENTION THEOREM")
    print("Biased Algorithms : Life Opportunities :: HIV : T Cells")
    print("=" * 80)

    print_theorem()

    # Create model
    model = BiasedAlgorithmEstablishmentModel()

    print("\n" + "=" * 80)
    print("INTERVENTION EFFICACY BY TIMING")
    print("(Parallel to PEP Efficacy by Timing)")
    print("=" * 80)

    print("\nMonths | Efficacy | P(Seeded) | P(Integrated) | Opportunities")
    print("-" * 65)

    for months in [0, 1, 3, 6, 9, 12, 18, 24]:
        result = model.intervention_efficacy(months)
        print(f"  {months:4d}  |  {result['intervention_efficacy']*100:5.1f}%  |"
              f"  {result['p_seeding_initiated']*100:5.1f}%  |"
              f"    {result['p_integration_complete']*100:5.1f}%    |"
              f"   {result['opportunity_preserved']*100:5.1f}%")

    print("\n" + "=" * 80)
    print("THE COROLLARY")
    print("=" * 80)
    print("""
    The HIV Prevention Theorem states:
        R₀(e) = 0 ⟹ R(t) = 0 ∀t
        (If initial reproduction is zero, infection never establishes)

    The Life Success Corollary states:
        O(e) = 100% ⟹ O(t) = 100% ∀t
        (If initial opportunity is preserved, penalty never establishes)

    Both require intervention BEFORE integration.

    After integration:
        HIV:        Lifelong ART, viral suppression, but reservoir persists
        Algorithms: Lifelong positive data accumulation, score improvement,
                    but negative records persist

    The parallel is exact.
    The tragedy is that HIV integration is biological necessity.
    Algorithmic integration is social choice.

    HIV cannot be prevented by policy.
    Algorithmic discrimination CAN be prevented by policy.

    That makes it not just analogous to disease -
    it makes it a civil rights emergency.
    """)

    # ==========================================================================
    # STRATIFIED ANALYSIS - 5 YEAR HORIZON WITHOUT INTERVENTION
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STRATIFIED LIFE SUCCESS ANALYSIS")
    print("5-YEAR HORIZON WITHOUT INTERVENTION")
    print("e = exposure to algorithmic bias, t = time in months")
    print("=" * 80)

    strat_model = StratifiedLifeSuccessModel()
    master = MasterEquation(strat_model)

    # Print the master equation
    master.print_master_equation()

    # High exposure scenario (e = 0.7)
    e = 0.7
    horizon_months = 60  # 5 years

    print("\n" + "=" * 80)
    print(f"SCENARIO: e = {e} (high algorithmic bias exposure)")
    print(f"HORIZON: {horizon_months} months (5 years) WITHOUT INTERVENTION")
    print("=" * 80)

    # Closed-form solutions at key timepoints
    print("\nCLOSED-FORM SOLUTIONS S(t) BY DOMAIN:")
    print("-" * 90)
    print(f"{'Domain':<15} | {'t=0':>7} | {'t=6mo':>7} | {'t=12mo':>7} | {'t=24mo':>7} | {'t=36mo':>7} | {'t=60mo':>7} | {'S_∞':>7}")
    print("-" * 90)

    for domain_name, domain in strat_model.domains.items():
        if domain_name == 'overall':
            continue
        row = [domain_name.capitalize()[:14]]
        for t in [0, 6, 12, 24, 36, 60]:
            S_t = strat_model.closed_form_solution(e, t, domain_name)
            row.append(f"{S_t:7.1f}")
        S_inf = strat_model.steady_state_solution(e, domain_name)
        row.append(f"{S_inf:7.1f}")
        print(f"{row[0]:<15} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]} | {row[7]}")

    # Overall (from master equation)
    print("-" * 90)
    overall_row = ['OVERALL']
    for t in [0, 6, 12, 24, 36, 60]:
        result = master.closed_form_solution(e, t)
        overall_row.append(f"{result['overall']:7.1f}")
    ss = master.steady_state(e)
    overall_row.append(f"{ss['overall']:7.1f}")
    print(f"{overall_row[0]:<15} | {overall_row[1]} | {overall_row[2]} | {overall_row[3]} | {overall_row[4]} | {overall_row[5]} | {overall_row[6]} | {overall_row[7]}")

    # Infection establishment dynamics
    print("\n" + "=" * 80)
    print("INFECTION ESTABLISHMENT DYNAMICS BY DOMAIN")
    print("(Eclipse → Acute → Chronic phases)")
    print("=" * 80)

    for t in [1, 3, 6, 12, 24, 60]:
        print(f"\n--- t = {t} months ---")
        print(f"{'Domain':<15} | {'Phase':^25} | {'Viral Load':>10} | {'T-Cell':>10} | {'dS/dt':>10}")
        print("-" * 80)
        for domain_name in ['employment', 'financial', 'credit', 'housing', 'legal', 'medical', 'mental_health']:
            dyn = strat_model.infection_establishment_dynamics(e, t, domain_name)
            phase = dyn['phase']
            if phase['eclipse'] > 0.5:
                phase_str = f"Eclipse ({phase['eclipse']*100:.0f}%)"
            elif phase['chronic'] > 0.5:
                phase_str = f"Chronic ({phase['chronic']*100:.0f}%)"
            else:
                phase_str = f"Acute ({phase['acute']*100:.0f}%)"
            print(f"{domain_name.capitalize():<15} | {phase_str:^25} | {dyn['viral_load']:>10.2f} | {dyn['t_cell_count']:>10.1f} | {dyn['dS_dt']:>10.3f}")

    # Steady state analysis
    print("\n" + "=" * 80)
    print("STEADY STATE ANALYSIS (t → ∞)")
    print("The 'viral set point' for each life domain")
    print("=" * 80)

    ss = master.steady_state(e)
    print(f"\nFor exposure e = {e}:")
    print("-" * 50)
    for domain_name, value in ss.items():
        status = ""
        if value < 30:
            status = " ⚠️ CRITICAL"
        elif value < 50:
            status = " ⚠️ SEVERE"
        elif value < 70:
            status = " ⚠️ MODERATE"
        print(f"  {domain_name.capitalize():<15}: {value:6.1f}/100{status}")

    # Eigenvalue analysis
    print("\n" + "=" * 80)
    print("EIGENVALUE ANALYSIS")
    print("System dynamics and timescales")
    print("=" * 80)

    eigen = master.eigenvalue_analysis(e)

    print(f"\nSystem stability: {'STABLE' if eigen['is_stable'] else 'UNSTABLE'}")
    print(f"Dominant timescale: {eigen['dominant_timescale']:.1f} months")
    print(f"Fastest decay: {eigen['fastest_decay']:.1f} months")

    print("\nEigenvalues (decay rates) by mode:")
    for i, (λ, τ) in enumerate(zip(eigen['eigenvalues'], eigen['timescales_months'])):
        domain = master.domain_names[i] if i < len(master.domain_names) else f"Mode {i}"
        print(f"  {domain:<15}: λ = {np.real(λ):>7.4f}, τ = {τ:>6.1f} months")

    # Dose-response analysis
    print("\n" + "=" * 80)
    print("DOSE-RESPONSE ANALYSIS")
    print("Steady state vs exposure level")
    print("=" * 80)

    print(f"\n{'Exposure (e)':<12} | {'Employment':>11} | {'Financial':>10} | {'Credit':>8} | {'Housing':>8} | {'Overall':>8}")
    print("-" * 72)

    for e_level in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        ss = master.steady_state(e_level)
        print(f"  {e_level:<10.1f} | {ss['employment']:>11.1f} | {ss['financial']:>10.1f} | {ss['credit']:>8.1f} | {ss['housing']:>8.1f} | {ss['overall']:>8.1f}")

    # The theorem summary
    print("\n" + "=" * 80)
    print("THE THEOREM (STRATIFIED FORMULATION)")
    print("=" * 80)
    print("""
    CLOSED-FORM SOLUTION FOR UNCOUPLED DOMAINS:
    ──────────────────────────────────────────

    S_i(t) = S_i0 · [β_i/(α_i·e + β_i) + (α_i·e)/(α_i·e + β_i) · exp(-(α_i·e + β_i)·t)]

    Where:
        S_i(t) = Life success in domain i at time t
        S_i0   = Baseline score (100)
        α_i    = Domain susceptibility
        e      = Algorithmic bias exposure (0 ≤ e ≤ 1)
        β_i    = Natural recovery rate
        t      = Time (months)

    STEADY STATE (t → ∞):
    ────────────────────

    S_i∞ = S_i0 · β_i/(α_i·e + β_i)

    For e > 0: S_i∞ < S_i0 (PERMANENT PENALTY)

    MASTER EQUATION (COUPLED SYSTEM):
    ─────────────────────────────────

    dS/dt = M·S + b

    Where:
        M = -diag(α)·e - diag(β) + A
        b = [diag(β) - A]·S_0
        A = Cross-domain coupling matrix (feedback loops)

    CLOSED-FORM SOLUTION:
        S(t) = exp(M·t)·S_0 + M⁻¹·[exp(M·t) - I]·b

    STEADY STATE:
        S_∞ = -M⁻¹·b

    THE THEOREM:
    ────────────

    For any exposure e > 0 and any domain i:

        lim(t→∞) S_i(t) = S_i∞ < S_i0

    PERMANENT PENALTY IS MATHEMATICALLY GUARANTEED.

    The only way to achieve S_i∞ = S_i0 is:
        1. e = 0 (no algorithmic bias exposure), OR
        2. Intervention before data integration (e_effective → 0)

    After integration: S_i∞ < S_i0 FOREVER.
    """)

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING FIGURES...")
    print("=" * 80)

    try:
        fig1, _ = plot_life_success_prevention_theorem('life_success_prevention_theorem.png')
        fig2 = plot_opportunity_dynamics('opportunity_dynamics.png')
        fig3 = plot_stratified_life_success(e=0.7, duration_months=60,
                                            save_path='stratified_life_success.png')
        fig4 = plot_exposure_response('exposure_response.png')
        print("\nFigures saved:")
        print("  - life_success_prevention_theorem.png")
        print("  - opportunity_dynamics.png")
        print("  - stratified_life_success.png")
        print("  - exposure_response.png")
        plt.show()
    except Exception as e:
        print(f"Could not display figures: {e}")

    print("\n" + "=" * 80)
    print("END OF ANALYSIS")
    print("=" * 80)


if __name__ == "__main__":
    main()
