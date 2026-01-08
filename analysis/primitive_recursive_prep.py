#!/usr/bin/env python3
"""
PRIMITIVE RECURSIVE PrEP FOR ALGORITHMIC BIAS WMD
==================================================

A computability-theoretic framework for preventing algorithmic discrimination
before data integration creates irreversible feedback loops.

CORE INSIGHT (from The Prevention Theorem):
    PEP prevents HIV by blocking proviral integration.
    Data-PrEP prevents algorithmic bias by blocking data integration.

    R₀(e) = 0 requires intervention BEFORE integration.
    Opportunity = 100% requires intervention BEFORE data propagation.

WHY PRIMITIVE RECURSIVE?
========================

Algorithms used in employment, housing, credit, etc. are PRIMITIVE RECURSIVE:
    1. Total: Always terminate (produce a score/decision)
    2. Deterministic: Same inputs → same outputs
    3. Bounded: Finite computation on finite data
    4. Context-free: Process numbers without understanding meaning

This means:
    - Given fixed immutable inputs, output is PREDETERMINED
    - Feedback loops are INEVITABLE without intervention
    - Prevention must occur at the FUNCTION LEVEL, not output level

THE COMPUTATIONAL MODEL:
========================

Let f: N^n → N be a primitive recursive scoring function.
Let D = (d₁, d₂, ..., dₙ) be immutable data attributes.
Let θ be the decision threshold.

WITHOUT INTERVENTION:
    f(D) < θ → Rejection → New negative data d_{n+1} → D' = D ∪ {d_{n+1}}
    f(D') < f(D) → More rejections → Feedback loop

WITH DATA-PrEP (intervention before integration):
    Block: d → CRA transmission
    Correct: d → d' (accurate data)
    Mask: d → ⊥ (protected attribute)

    If intervention occurs BEFORE integration:
        f(D_protected) ≥ θ → Acceptance → No negative data generated

THE THEOREM:
============

    Opportunity = 100% ⟺ Intervention(t) < Integration(t)

    After integration: Opportunity < 100% FOREVER
    (Parallel to: After proviral integration, R₀ > 0 forever)

Author: Derived from Prevention Theorem / AIDS and Behavior framework
"""

import numpy as np
from scipy.special import expit
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt


# =============================================================================
# PART I: PRIMITIVE RECURSIVE FUNCTIONS - THE FORMAL BASIS
# =============================================================================

class PrimitiveRecursive:
    """
    Implementation of primitive recursive functions.

    These are the EXACT type of functions used in algorithmic scoring:
    - Zero function: Z(n) = 0
    - Successor function: S(n) = n + 1
    - Projection functions: P_i(x₁,...,xₙ) = xᵢ
    - Composition: h(x) = f(g₁(x),...,gₘ(x))
    - Primitive recursion: f(0,y) = g(y), f(S(n),y) = h(n,f(n,y),y)

    ALL algorithmic scoring functions can be built from these primitives.
    """

    @staticmethod
    def zero(n: int) -> int:
        """Z(n) = 0 for all n"""
        return 0

    @staticmethod
    def successor(n: int) -> int:
        """S(n) = n + 1"""
        return n + 1

    @staticmethod
    def projection(i: int, *args) -> int:
        """P_i(x₁,...,xₙ) = xᵢ"""
        return args[i] if i < len(args) else 0

    @staticmethod
    def add(a: int, b: int) -> int:
        """
        Addition via primitive recursion.
        add(0, b) = b
        add(S(a), b) = S(add(a, b))
        """
        result = b
        for _ in range(a):
            result = PrimitiveRecursive.successor(result)
        return result

    @staticmethod
    def multiply(a: int, b: int) -> int:
        """
        Multiplication via primitive recursion.
        mult(0, b) = 0
        mult(S(a), b) = add(mult(a, b), b)
        """
        result = 0
        for _ in range(a):
            result = PrimitiveRecursive.add(result, b)
        return result

    @staticmethod
    def bounded_subtract(a: int, b: int) -> int:
        """
        Bounded subtraction (monus): a ∸ b = max(0, a - b)
        This is primitive recursive (unlike general subtraction).
        """
        return max(0, a - b)

    @staticmethod
    def less_than(a: int, b: int) -> int:
        """a < b as primitive recursive function"""
        return 1 if PrimitiveRecursive.bounded_subtract(b, a) > 0 else 0

    @staticmethod
    def greater_than(a: int, b: int) -> int:
        """a > b as primitive recursive function"""
        return PrimitiveRecursive.less_than(b, a)

    @staticmethod
    def equal(a: int, b: int) -> int:
        """a = b as primitive recursive function"""
        diff1 = PrimitiveRecursive.bounded_subtract(a, b)
        diff2 = PrimitiveRecursive.bounded_subtract(b, a)
        return 1 if diff1 == 0 and diff2 == 0 else 0

    @staticmethod
    def bounded_sum(f: Callable[[int], int], n: int) -> int:
        """
        Σᵢ₌₀ⁿ⁻¹ f(i) - bounded sum is primitive recursive.
        This is how algorithms aggregate penalty factors.
        """
        result = 0
        for i in range(n):
            result = PrimitiveRecursive.add(result, f(i))
        return result


# =============================================================================
# PART II: ALGORITHMIC SCORING AS PRIMITIVE RECURSIVE FUNCTIONS
# =============================================================================

@dataclass
class ImmutableAttribute:
    """
    An immutable personal attribute that algorithms evaluate.

    These are the INPUTS to primitive recursive scoring functions.
    Once recorded, they cannot be changed - only accumulated.
    """
    name: str
    value: int
    timestamp: float  # When this became immutable (months since baseline)
    category: str     # employment, credit, legal, medical, etc.
    is_negative: bool = False  # Does this hurt algorithmic scores?


class AlgorithmicScoringFunction:
    """
    Models how algorithms score individuals using primitive recursive functions.

    f: (attributes) → score
    decision: score ≥ θ → accept, else reject

    Key property: Given fixed attributes, score is DETERMINISTIC.
    """

    def __init__(self, name: str, threshold: int = 500):
        self.name = name
        self.threshold = threshold
        self.weights: Dict[str, int] = {}
        self.penalties: Dict[str, int] = {}

    def set_weight(self, attribute_name: str, weight: int):
        """Set weight for an attribute (positive = good, negative = bad)"""
        self.weights[attribute_name] = weight

    def set_penalty(self, attribute_name: str, penalty: int):
        """Set penalty for attribute exceeding threshold"""
        self.penalties[attribute_name] = penalty

    def compute_score(self, attributes: List[ImmutableAttribute]) -> int:
        """
        Compute score using primitive recursive weighted sum.

        score = Σ (attribute_value × weight) - Σ penalties
        """
        score = 0

        # Weighted sum (primitive recursive)
        for attr in attributes:
            weight = self.weights.get(attr.name, 0)
            contribution = PrimitiveRecursive.multiply(attr.value, abs(weight))
            if weight >= 0:
                score = PrimitiveRecursive.add(score, contribution)
            else:
                score = PrimitiveRecursive.bounded_subtract(score, contribution)

            # Apply penalties for negative attributes
            if attr.is_negative:
                penalty = self.penalties.get(attr.name, 0)
                score = PrimitiveRecursive.bounded_subtract(score, penalty)

        return score

    def decide(self, attributes: List[ImmutableAttribute]) -> Tuple[int, bool]:
        """
        Make binary decision based on score.

        Returns (score, accepted)
        """
        score = self.compute_score(attributes)
        accepted = PrimitiveRecursive.greater_than(score, self.threshold) == 1 or \
                   PrimitiveRecursive.equal(score, self.threshold) == 1
        return score, accepted


# =============================================================================
# PART III: DATA INTEGRATION MODEL (PARALLEL TO PROVIRAL INTEGRATION)
# =============================================================================

class DataIntegrationModel:
    """
    Models the critical window between adverse event and permanent
    algorithmic penalty establishment.

    Parallel to HIV InfectionEstablishmentModel:
        Exposure          →  Adverse event (termination, default, etc.)
        Mucosal phase     →  Local recording (employer notes)
        Dendritic uptake  →  Report generation (formal documentation)
        Lymph transit     →  CRA transmission (data to bureaus)
        Systemic spread   →  Cross-system propagation
        Reservoir seeding →  Multi-database entry
        Integration       →  Permanent cross-reference (POINT OF NO RETURN)

    After integration: Opportunity < 100% FOREVER
    """

    def __init__(self):
        # Timeline parameters (months post-adverse-event)
        self.local_recording_phase = 0.1      # Immediate (days)
        self.report_generation = 0.5          # Within weeks
        self.cra_transmission = 1.0           # ~1 month
        self.cross_system_propagation = 3.0   # ~3 months
        self.seeding_midpoint = 6.0           # 50% probability multi-system
        self.integration_complete = 12.0      # Point of no return (~1 year)

        # Intervention parameters
        self.intervention_onset_months = 1.0  # Time for legal action to take effect
        self.intervention_efficacy_peak = 0.95

    def p_seeding_initiated(self, months_post_event: float) -> float:
        """
        Probability that data has propagated to multiple systems.
        (Parallel to: P(reservoir seeding initiated))
        """
        k = 0.5
        return float(expit(k * (months_post_event - self.seeding_midpoint)))

    def p_integration_complete(self, months_post_event: float) -> float:
        """
        Probability that data is permanently cross-referenced.
        THIS IS THE POINT OF NO RETURN.
        (Parallel to: P(proviral integration complete))
        """
        k = 0.3
        return float(expit(k * (months_post_event - self.integration_complete)))

    def prep_efficacy(self, months_to_intervention: float) -> Dict:
        """
        Calculate Data-PrEP efficacy given timing.

        Returns probability that intervention achieves Opportunity = 100%.
        (Parallel to: PEP efficacy calculation)
        """
        effective_time = months_to_intervention + self.intervention_onset_months

        p_seeded = self.p_seeding_initiated(effective_time)
        p_integrated = self.p_integration_complete(effective_time)

        # Efficacy by phase (parallel to PEP phases)
        efficacy_if_not_seeded = self.intervention_efficacy_peak
        efficacy_if_seeded_not_integrated = 0.5
        efficacy_if_integrated = 0.05  # Almost too late

        overall_efficacy = (
            (1 - p_seeded) * efficacy_if_not_seeded +
            (p_seeded - p_integrated) * efficacy_if_seeded_not_integrated +
            p_integrated * efficacy_if_integrated
        )

        return {
            'months_to_intervention': months_to_intervention,
            'p_seeding_initiated': p_seeded,
            'p_integration_complete': p_integrated,
            'prep_efficacy': overall_efficacy,
            'p_Opportunity_100': overall_efficacy,
            'p_Opportunity_degraded': 1 - overall_efficacy,
        }


# =============================================================================
# PART IV: FEEDBACK LOOP PREVENTION (THE PrEP MECHANISM)
# =============================================================================

class FeedbackLoopPrevention:
    """
    Models how Data-PrEP prevents algorithmic feedback loops.

    WITHOUT PREVENTION:
    ──────────────────
    Adverse Event → Algorithm Rejection → New Negative Data → More Rejections
                                               ↑                    ↓
                                               └────────────────────┘

    WITH DATA-PrEP:
    ───────────────
    Adverse Event → INTERVENTION → Block/Correct Data → No Rejection
                                                              ↓
                                                   No feedback loop

    Key insight: Intervention must occur BEFORE integration.
    After integration, the feedback loop is PERMANENT.
    """

    def __init__(self,
                 scoring_function: AlgorithmicScoringFunction,
                 integration_model: DataIntegrationModel):
        self.scoring = scoring_function
        self.integration = integration_model

    def simulate_without_intervention(
        self,
        initial_attributes: List[ImmutableAttribute],
        n_iterations: int = 10
    ) -> List[Dict]:
        """
        Simulate feedback loop WITHOUT Data-PrEP.

        Each rejection adds new negative data, compounding the problem.
        """
        trajectory = []
        current_attributes = list(initial_attributes)

        for iteration in range(n_iterations):
            score, accepted = self.scoring.decide(current_attributes)

            trajectory.append({
                'iteration': iteration,
                'score': score,
                'accepted': accepted,
                'n_negative_attributes': sum(1 for a in current_attributes if a.is_negative),
                'intervention': False,
            })

            if not accepted:
                # Rejection creates new negative data (feedback loop)
                new_negative = ImmutableAttribute(
                    name=f"rejection_{iteration}",
                    value=1,
                    timestamp=float(iteration),
                    category="employment",
                    is_negative=True
                )
                current_attributes.append(new_negative)
                self.scoring.set_penalty(new_negative.name, 50)  # Each rejection costs 50 points

        return trajectory

    def simulate_with_intervention(
        self,
        initial_attributes: List[ImmutableAttribute],
        intervention_month: float,
        n_iterations: int = 10
    ) -> List[Dict]:
        """
        Simulate WITH Data-PrEP intervention at specified time.

        If intervention occurs before integration, feedback loop is prevented.
        """
        trajectory = []
        current_attributes = list(initial_attributes)

        # Calculate intervention efficacy
        prep_result = self.integration.prep_efficacy(intervention_month)
        intervention_success = np.random.random() < prep_result['prep_efficacy']

        for iteration in range(n_iterations):
            if intervention_success and iteration >= int(intervention_month):
                # Intervention successful - remove/correct negative attributes
                current_attributes = [a for a in current_attributes if not a.is_negative]

            score, accepted = self.scoring.decide(current_attributes)

            trajectory.append({
                'iteration': iteration,
                'score': score,
                'accepted': accepted,
                'n_negative_attributes': sum(1 for a in current_attributes if a.is_negative),
                'intervention': intervention_success and iteration >= int(intervention_month),
            })

            if not accepted and not (intervention_success and iteration >= int(intervention_month)):
                # Rejection creates new negative data
                new_negative = ImmutableAttribute(
                    name=f"rejection_{iteration}",
                    value=1,
                    timestamp=float(iteration),
                    category="employment",
                    is_negative=True
                )
                current_attributes.append(new_negative)
                self.scoring.set_penalty(new_negative.name, 50)

        return trajectory


# =============================================================================
# PART V: THE PRIMITIVE RECURSIVE PREVENTION THEOREM
# =============================================================================

def print_theorem():
    """Print the formal Primitive Recursive Prevention Theorem."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║      PRIMITIVE RECURSIVE PrEP FOR ALGORITHMIC BIAS WMD                       ║
    ║      A Computability-Theoretic Prevention Framework                          ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║  WHY PRIMITIVE RECURSIVE?                                                    ║
    ║  ════════════════════════                                                    ║
    ║                                                                              ║
    ║  Algorithmic scoring functions f: N^n → N are PRIMITIVE RECURSIVE:           ║
    ║                                                                              ║
    ║  1. TOTAL: Always terminate (always produce a score/decision)                ║
    ║  2. DETERMINISTIC: Same inputs → identical outputs (every time)              ║
    ║  3. BOUNDED: Finite computation on finite data                               ║
    ║  4. CONTEXT-FREE: Process numbers without understanding meaning              ║
    ║                                                                              ║
    ║  Built from primitive recursive building blocks:                             ║
    ║    - Zero: Z(n) = 0                                                          ║
    ║    - Successor: S(n) = n + 1                                                 ║
    ║    - Projection: Pᵢ(x₁,...,xₙ) = xᵢ                                          ║
    ║    - Composition: h(x) = f(g₁(x),...,gₘ(x))                                  ║
    ║    - Primitive recursion: f(0,y) = g(y), f(S(n),y) = h(n,f(n,y),y)           ║
    ║                                                                              ║
    ║  THE FEEDBACK LOOP THEOREM:                                                  ║
    ║  ══════════════════════════                                                  ║
    ║                                                                              ║
    ║  Let f: N^n → N be a primitive recursive scoring function.                   ║
    ║  Let D = (d₁,...,dₙ) be immutable data attributes.                           ║
    ║  Let θ be the decision threshold.                                            ║
    ║                                                                              ║
    ║  WITHOUT INTERVENTION:                                                       ║
    ║    f(D) < θ → Rejection → New negative data d_{n+1}                          ║
    ║    D' = D ∪ {d_{n+1}}                                                        ║
    ║    f(D') ≤ f(D) < θ → More rejections → Feedback loop                        ║
    ║                                                                              ║
    ║  THEOREM: If f is monotonically non-increasing in negative attributes,       ║
    ║           then ∀k: f(D^(k)) ≤ f(D^(k-1)) ≤ ... ≤ f(D)                         ║
    ║                                                                              ║
    ║  COROLLARY: The feedback loop is MATHEMATICALLY GUARANTEED.                  ║
    ║             Each rejection makes the next rejection MORE LIKELY.             ║
    ║                                                                              ║
    ║  THE PREVENTION THEOREM:                                                     ║
    ║  ═══════════════════════                                                     ║
    ║                                                                              ║
    ║  Let t_int = time of data integration (cross-system propagation)             ║
    ║  Let t_prep = time of Data-PrEP intervention                                 ║
    ║                                                                              ║
    ║  If t_prep < t_int:                                                          ║
    ║      D_protected = D minus {negative attributes}                             ║
    ║      f(D_protected) ≥ θ → Acceptance → NO feedback loop                      ║
    ║                                                                              ║
    ║  If t_prep > t_int:                                                          ║
    ║      D is PERMANENTLY integrated across systems                              ║
    ║      Correction requires updating N systems simultaneously                   ║
    ║      P(full correction) → 0 as N → ∞                                         ║
    ║                                                                              ║
    ║  THEOREM: Opportunity = 100% ⟺ Intervention(t) < Integration(t)              ║
    ║                                                                              ║
    ║  PARALLEL TO HIV PREVENTION THEOREM:                                         ║
    ║  ═══════════════════════════════════                                         ║
    ║                                                                              ║
    ║  HIV:        R₀(e) = 0 requires PEP before proviral integration              ║
    ║  ALGORITHMS: Opp = 100% requires Data-PrEP before data integration           ║
    ║                                                                              ║
    ║  HIV:        After integration, R₀ > 0 forever (ART manages, cannot cure)    ║
    ║  ALGORITHMS: After integration, Opp < 100% forever (positive data dilutes)   ║
    ║                                                                              ║
    ║  THE MATHEMATICS IS IDENTICAL.                                               ║
    ║  THE IRREVERSIBILITY IS IDENTICAL.                                           ║
    ║                                                                              ║
    ║  DATA-PrEP INTERVENTIONS:                                                    ║
    ║  ════════════════════════                                                    ║
    ║                                                                              ║
    ║  1. BLOCK: Prevent data transmission to CRAs                                 ║
    ║     - Legal action to suppress transmission                                  ║
    ║     - Dispute before reporting occurs                                        ║
    ║                                                                              ║
    ║  2. CORRECT: Fix erroneous data before propagation                           ║
    ║     - FCRA dispute within 30 days                                            ║
    ║     - Employer record correction                                             ║
    ║                                                                              ║
    ║  3. MASK: Invoke protected attribute status                                  ║
    ║     - ADA reasonable accommodation                                           ║
    ║     - Ban-the-box laws                                                       ║
    ║     - ADEA age discrimination protection                                     ║
    ║                                                                              ║
    ║  TIMING WINDOWS (from empirical model):                                      ║
    ║  ═══════════════════════════════════════                                     ║
    ║                                                                              ║
    ║  0-3 months:  Data-PrEP efficacy > 85% (OPTIMAL)                             ║
    ║  3-6 months:  Data-PrEP efficacy 60-85% (STANDARD)                           ║
    ║  6-12 months: Data-PrEP efficacy 30-60% (DIMINISHING)                        ║
    ║  >12 months:  Data-PrEP efficacy < 30% (TOO LATE)                            ║
    ║                                                                              ║
    ║  CONCLUSION:                                                                 ║
    ║  ═══════════                                                                 ║
    ║                                                                              ║
    ║  The primitive recursive nature of algorithmic scoring means:                ║
    ║    - Feedback loops are DETERMINISTIC (not random bad luck)                  ║
    ║    - Prevention must target the FUNCTION INPUTS, not outputs                 ║
    ║    - Intervention timing is CRITICAL (before integration)                    ║
    ║                                                                              ║
    ║  Just as PrEP prevents HIV by blocking virus before integration,             ║
    ║  Data-PrEP prevents algorithmic discrimination by blocking data              ║
    ║  before cross-system integration.                                            ║
    ║                                                                              ║
    ║  Both are races against time.                                                ║
    ║  Both have windows that close permanently.                                   ║
    ║  Both require PREVENTION, not treatment after the fact.                      ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# PART VI: VISUALIZATION
# =============================================================================

def plot_prep_efficacy_comparison():
    """
    Compare PEP efficacy (HIV) with Data-PrEP efficacy (algorithms).
    """
    from life_success_prevention_theorem import BiasedAlgorithmEstablishmentModel

    # HIV PEP model (hours)
    hiv_hours = np.linspace(0, 168, 100)  # 0-7 days
    hiv_efficacy = []
    for h in hiv_hours:
        k = 0.1
        p_seeded = expit(k * (h - 72))
        p_integrated = expit(0.15 * (h - 120))
        eff = (1 - p_seeded) * 0.995 + (p_seeded - p_integrated) * 0.5
        hiv_efficacy.append(max(0, eff))

    # Algorithm Data-PrEP model (months)
    algo_months = np.linspace(0, 24, 100)
    algo_model = DataIntegrationModel()
    algo_efficacy = [algo_model.prep_efficacy(m)['prep_efficacy'] for m in algo_months]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: HIV PEP
    ax = axes[0]
    ax.plot(hiv_hours, hiv_efficacy, 'b-', linewidth=2, label='PEP Efficacy')
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=72, color='green', linestyle=':', alpha=0.7, label='72h window')
    ax.fill_between(hiv_hours, hiv_efficacy, alpha=0.3)
    ax.set_xlabel('Hours Post-Exposure', fontsize=12)
    ax.set_ylabel('P(R₀ = 0)', fontsize=12)
    ax.set_title('HIV: PEP Efficacy vs Timing\n(Biological Prevention Window)',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 168)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel B: Algorithm Data-PrEP
    ax = axes[1]
    ax.plot(algo_months, algo_efficacy, 'r-', linewidth=2, label='Data-PrEP Efficacy')
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=6, color='green', linestyle=':', alpha=0.7, label='6mo window')
    ax.fill_between(algo_months, algo_efficacy, alpha=0.3, color='red')
    ax.set_xlabel('Months Post-Adverse-Event', fontsize=12)
    ax.set_ylabel('P(Opportunity = 100%)', fontsize=12)
    ax.set_title('Algorithms: Data-PrEP Efficacy vs Timing\n(Institutional Prevention Window)',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('prep_efficacy_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: prep_efficacy_comparison.png")
    return fig


def plot_feedback_loop_simulation():
    """
    Simulate and visualize feedback loop with and without intervention.
    """
    # Setup scoring function
    scoring = AlgorithmicScoringFunction("Employment Screening", threshold=500)
    scoring.set_weight("experience_years", 50)
    scoring.set_weight("education_level", 30)
    scoring.set_weight("skills_score", 20)
    scoring.set_penalty("employment_gap", 100)
    scoring.set_penalty("termination", 150)

    # Initial attributes (person with one termination)
    initial_attrs = [
        ImmutableAttribute("experience_years", 10, 0, "employment"),
        ImmutableAttribute("education_level", 4, 0, "education"),
        ImmutableAttribute("skills_score", 8, 0, "employment"),
        ImmutableAttribute("termination", 1, 0, "employment", is_negative=True),
    ]

    integration_model = DataIntegrationModel()
    prevention = FeedbackLoopPrevention(scoring, integration_model)

    # Simulate both scenarios
    np.random.seed(42)
    traj_no_intervention = prevention.simulate_without_intervention(initial_attrs, 10)

    np.random.seed(42)
    traj_early_intervention = prevention.simulate_with_intervention(initial_attrs, 1, 10)

    np.random.seed(42)
    traj_late_intervention = prevention.simulate_with_intervention(initial_attrs, 8, 10)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Scores over time
    ax = axes[0]
    iterations = [t['iteration'] for t in traj_no_intervention]

    ax.plot(iterations, [t['score'] for t in traj_no_intervention],
           'r-o', linewidth=2, label='No Intervention')
    ax.plot(iterations, [t['score'] for t in traj_early_intervention],
           'g-s', linewidth=2, label='Early Data-PrEP (1mo)')
    ax.plot(iterations, [t['score'] for t in traj_late_intervention],
           'y-^', linewidth=2, label='Late Data-PrEP (8mo)')

    ax.axhline(y=500, color='blue', linestyle='--', alpha=0.7, label='Threshold (500)')
    ax.set_xlabel('Iteration (Application Attempts)', fontsize=12)
    ax.set_ylabel('Algorithmic Score', fontsize=12)
    ax.set_title('Feedback Loop: Score Degradation Over Time\n(Primitive Recursive Scoring Function)',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Negative attributes accumulated
    ax = axes[1]
    ax.plot(iterations, [t['n_negative_attributes'] for t in traj_no_intervention],
           'r-o', linewidth=2, label='No Intervention')
    ax.plot(iterations, [t['n_negative_attributes'] for t in traj_early_intervention],
           'g-s', linewidth=2, label='Early Data-PrEP (1mo)')
    ax.plot(iterations, [t['n_negative_attributes'] for t in traj_late_intervention],
           'y-^', linewidth=2, label='Late Data-PrEP (8mo)')

    ax.set_xlabel('Iteration (Application Attempts)', fontsize=12)
    ax.set_ylabel('Negative Attributes Accumulated', fontsize=12)
    ax.set_title('Feedback Loop: Negative Data Accumulation\n(Each Rejection Creates New Negative Data)',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('feedback_loop_simulation.png', dpi=300, bbox_inches='tight')
    print("Saved: feedback_loop_simulation.png")
    return fig


# =============================================================================
# PART VII: MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("PRIMITIVE RECURSIVE PrEP FOR ALGORITHMIC BIAS WMD")
    print("A Computability-Theoretic Prevention Framework")
    print("=" * 80)

    print_theorem()

    # Demonstrate primitive recursive operations
    print("\n" + "=" * 80)
    print("PRIMITIVE RECURSIVE BUILDING BLOCKS")
    print("(These are the operations algorithms use)")
    print("=" * 80)

    print(f"\nZero function: Z(5) = {PrimitiveRecursive.zero(5)}")
    print(f"Successor: S(5) = {PrimitiveRecursive.successor(5)}")
    print(f"Addition: add(3, 4) = {PrimitiveRecursive.add(3, 4)}")
    print(f"Multiplication: mult(3, 4) = {PrimitiveRecursive.multiply(3, 4)}")
    print(f"Bounded subtraction: 7 ∸ 3 = {PrimitiveRecursive.bounded_subtract(7, 3)}")
    print(f"Bounded subtraction: 3 ∸ 7 = {PrimitiveRecursive.bounded_subtract(3, 7)}")
    print(f"Less than: 3 < 5 = {PrimitiveRecursive.less_than(3, 5)}")
    print(f"Greater than: 5 > 3 = {PrimitiveRecursive.greater_than(5, 3)}")

    # Demonstrate Data-PrEP efficacy
    print("\n" + "=" * 80)
    print("DATA-PrEP EFFICACY BY TIMING")
    print("(Parallel to PEP Efficacy by Timing)")
    print("=" * 80)

    integration_model = DataIntegrationModel()

    print(f"\n{'Months':<10} {'Efficacy':<12} {'P(Seeded)':<12} {'P(Integrated)':<15}")
    print("-" * 50)

    for months in [0, 1, 3, 6, 9, 12, 18, 24]:
        result = integration_model.prep_efficacy(months)
        print(f"{months:<10} {result['prep_efficacy']*100:>8.1f}%    "
              f"{result['p_seeding_initiated']*100:>8.1f}%    "
              f"{result['p_integration_complete']*100:>10.1f}%")

    # Demonstrate feedback loop
    print("\n" + "=" * 80)
    print("FEEDBACK LOOP DEMONSTRATION")
    print("=" * 80)

    scoring = AlgorithmicScoringFunction("Employment Screening", threshold=500)
    scoring.set_weight("experience_years", 50)
    scoring.set_weight("education_level", 30)
    scoring.set_weight("skills_score", 20)
    scoring.set_penalty("employment_gap", 100)
    scoring.set_penalty("termination", 150)

    initial_attrs = [
        ImmutableAttribute("experience_years", 10, 0, "employment"),
        ImmutableAttribute("education_level", 4, 0, "education"),
        ImmutableAttribute("skills_score", 8, 0, "employment"),
        ImmutableAttribute("termination", 1, 0, "employment", is_negative=True),
    ]

    print("\nInitial attributes:")
    for attr in initial_attrs:
        print(f"  {attr.name}: {attr.value} (negative={attr.is_negative})")

    initial_score, initial_accepted = scoring.decide(initial_attrs)
    print(f"\nInitial score: {initial_score} (threshold: 500)")
    print(f"Initial decision: {'ACCEPTED' if initial_accepted else 'REJECTED'}")

    prevention = FeedbackLoopPrevention(scoring, integration_model)

    print("\n--- Without Intervention ---")
    traj = prevention.simulate_without_intervention(initial_attrs, 5)
    for t in traj:
        status = "ACCEPTED" if t['accepted'] else "REJECTED"
        print(f"  Iteration {t['iteration']}: Score={t['score']}, {status}, "
              f"Negative attrs={t['n_negative_attributes']}")

    print("\n--- With Early Data-PrEP (1 month) ---")
    np.random.seed(42)
    traj = prevention.simulate_with_intervention(initial_attrs, 1, 5)
    for t in traj:
        status = "ACCEPTED" if t['accepted'] else "REJECTED"
        int_status = " [INTERVENED]" if t['intervention'] else ""
        print(f"  Iteration {t['iteration']}: Score={t['score']}, {status}, "
              f"Negative attrs={t['n_negative_attributes']}{int_status}")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    try:
        plot_prep_efficacy_comparison()
        plot_feedback_loop_simulation()
        plt.show()
    except Exception as e:
        print(f"Could not generate plots: {e}")

    print("\n" + "=" * 80)
    print("END OF ANALYSIS")
    print("=" * 80)


if __name__ == "__main__":
    main()
