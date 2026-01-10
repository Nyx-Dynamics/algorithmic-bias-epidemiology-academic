# Why Incremental AI Governance Fails: A Systems-Failure Analysis of Algorithmic Harm

**Policy White Paper**

Prepared for: Consumer Financial Protection Bureau, Department of Health and Human Services, Equal Employment Opportunity Commission, Department of Justice Civil Rights Division, CMS Innovation Center

**Author:** AC Demidont, DO | Nyx Dynamics LLC

**Date:** January 2026

**Based on:** Algorithmic Bias Epidemiology: A Computational Framework for Modeling Algorithmic Discrimination as an Epidemiological Process (2026)

---

## Executive Summary

Algorithmic systems now determine access to employment, credit, housing, and healthcare for most Americans. Despite decades of regulatory effort targeting individual aspects of algorithmic harm, discrimination persists at population scale. This paper explains why.

Using validated mathematical modeling, we demonstrate that algorithmic discrimination operates as a **synergistic system** where barriers reinforce each other across institutional boundaries. The dominant finding: **87.6% of harm arises from the interaction between barrier layers**, not from any individual barrier.

This result is non-negotiable. It does not depend on intent, technical accuracy, or the sophistication of any single algorithm. It is a structural property of interconnected systems.

### Key Graphic: Interaction Dominance

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   CONTRIBUTION TO ALGORITHMIC HARM                              │
│                                                                 │
│   Individual Barriers (addressable by single agency)     0.3%  │
│   ████                                                          │
│                                                                 │
│   Two-Way Interactions (addressable by paired agencies) 11.9%  │
│   ██████████████████████████████████████                        │
│                                                                 │
│   Three-Way Interaction (requires full coordination)    87.6%  │
│   ████████████████████████████████████████████████████████████  │
│   ████████████████████████████████████████████████████████████  │
│   ████████████████████████████████████████████████████████████  │
│   ████████████████████████████████████████████████████████████  │
│   ██████████████████████████████████████████████████████        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Table: Cost of Partial vs. Comprehensive Reform

| Reform Strategy | Success Rate | Cost per 1% Improvement |
|-----------------|--------------|-------------------------|
| Single barrier removal | <0.02% | $50,000+ |
| Single agency action | <1% | $15,000+ |
| Paired agency coordination | 11% | $1,200 |
| **Full coordination** | **95%** | **$142** |

### Conclusion

**Fragmented oversight guarantees persistent harm, independent of intent or technical accuracy.**

---

## 1. The Problem Regulators Think They Are Solving

Current regulatory frameworks treat algorithmic discrimination as three separate problems:

### Bias as Error

Regulators assume discrimination results from technical mistakes—flawed training data, biased variables, or miscalibrated models. The solution: require audits, mandate bias testing, demand documentation.

**Reality:** Our analysis shows that even with perfectly accurate algorithms, the system produces a baseline success rate of 0.0018%. Fewer than 2 in 100,000 individuals can successfully resolve algorithmic harm. Technical accuracy is irrelevant to this outcome.

### Fairness as Calibration

Regulators assume that if individual algorithms meet fairness metrics, the system will be fair. The solution: define protected classes, establish disparate impact thresholds, require fairness reporting.

**Reality:** Fairness at the component level does not produce fairness at the system level. An individual can face a "fair" credit algorithm, a "fair" employment screen, and a "fair" healthcare allocation model—and still experience compounding harm that no single agency can detect or address.

### Oversight as Siloed Compliance

Each agency enforces its own statute:
- CFPB enforces FCRA and ECOA for credit
- EEOC enforces Title VII for employment
- HHS enforces civil rights provisions for healthcare
- HUD enforces FHA for housing

**Reality:** Algorithmic systems do not respect statutory boundaries. A criminal record affects employment screening, which affects credit scores, which affects housing access, which affects healthcare utilization. Each agency sees only its fragment. No agency sees the cascade.

---

## 2. The Problem the System Is Actually Solving

From a systems perspective, the current algorithmic infrastructure solves three problems—none of which involve fairness:

### Risk Externalization

Algorithmic systems allow institutions to transfer decision-making risk to automated processes. When harm occurs, institutions point to the algorithm. When the algorithm fails, institutions point to the data. When the data is wrong, institutions point to the source. Accountability disperses until it disappears.

**Quantitative finding:** In our model, successful resolution requires navigating 11 distinct barriers across 3 institutional layers. Each barrier represents a point where accountability can be deflected.

### Information Persistence

Once adverse information enters interconnected data systems, it propagates indefinitely. A single error—a misidentification, a disputed debt, a false arrest—replicates across credit bureaus, background check databases, healthcare risk scores, and fraud detection systems.

**Quantitative finding:** The Data Integration layer alone (3 barriers) reduces success probability to 2.6%. Combined with Data Accuracy barriers, success drops to 0.4%. This persistence is structural, not accidental.

### Institutional Diffusion of Accountability

No single institution owns the problem:
- Credit bureaus compile data but don't make decisions
- Lenders make decisions but don't control data
- Employers use screening services but don't audit them
- Healthcare systems implement algorithms they didn't develop

**Quantitative finding:** The dominant three-way interaction (87.6%) exists precisely because responsibility is distributed across layers that do not coordinate.

---

## 3. The Mathematical Result (Non-Negotiable)

### What We Measured

We modeled algorithmic discrimination as an 11-barrier system across three layers:

**Layer 1 - Data Integration:** How adverse information spreads across systems
- Cross-system data sharing
- Multi-database flagging
- Rapid data transmission

**Layer 2 - Data Accuracy:** How errors persist and compound
- Error correction difficulty
- Identity verification complexity
- Systemic algorithmic bias

**Layer 3 - Institutional Response:** How individuals can (or cannot) seek remedy
- Legal knowledge requirements
- Financial resources for advocacy
- Time constraints for disputes
- Retaliation concerns
- Procedural complexity

### What We Found

**Finding 1: Individual barrier removal produces negligible improvement**

Removing any single barrier while others remain produces less than 0.02% improvement in success probability. This is not a small effect—it is effectively zero.

This finding explains why:
- Improving credit report dispute processes has not reduced credit discrimination
- Banning specific hiring algorithms has not reduced employment discrimination
- Requiring bias audits has not reduced healthcare discrimination

**Finding 2: The order of reform does not matter**

We tested five different reform strategies:
- Forward (address data problems first, then accuracy, then institutions)
- Backward (address institutions first, then accuracy, then data)
- Greedy by impact (address largest barriers first)
- Greedy by cost (address cheapest barriers first)
- Random ordering

All strategies produced statistically identical outcomes (ANOVA: F=0.23, p=0.92).

This finding explains why pilot programs fail. Whether you start with "quick wins" or "big problems," the trajectory is the same: near-zero improvement until nearly all barriers are addressed.

**Finding 3: 87.6% of the problem requires addressing all three layers simultaneously**

This is not a policy preference. It is a mathematical property of multiplicative barrier systems.

When barriers operate independently, removing one provides proportional benefit. When barriers interact synergistically, removing one provides almost no benefit because the remaining barriers still block success.

The three-way interaction dominance (87.6%) means that:
- Agency-specific solutions cannot work by design
- Sequenced reforms cannot work by design
- Partial coordination cannot work by design

### Robustness

These findings are not artifacts of our parameter choices:
- 100% of 1,000 bootstrap samples confirmed three-way interaction dominance
- Results remained stable under up to 25% parameter uncertainty
- Four independent sensitivity methods (OAT, Sobol, Morris, bootstrap) produced consistent conclusions

---

## 4. Regulatory Implications (Actionable but Uncomfortable)

### Why Agency-Specific Rulemaking Cannot Succeed Alone

The CFPB can require credit bureaus to improve dispute processes. This addresses one barrier.

The EEOC can require employers to validate screening algorithms. This addresses one barrier.

HHS can require healthcare systems to audit risk scores. This addresses one barrier.

Each agency can fully achieve its regulatory objectives and still produce zero population-level improvement. This is not a failure of implementation. It is a structural impossibility.

**Implication:** Regulatory success cannot be measured by agency-level compliance metrics. It can only be measured by system-level outcomes.

### Why Sequencing Reforms Is Irrelevant

Policy discussions often focus on where to start:
- "Fix the data first, then address institutional barriers"
- "Empower individuals first, then address technical problems"
- "Start with healthcare because lives are at stake"

Our analysis proves these debates are irrelevant. All sequences produce the same trajectory: near-zero improvement until comprehensive action is taken.

**Implication:** Time spent debating priorities is time wasted. The only relevant question is whether coordination will occur.

### Why Enforcement Without Coordination Amplifies Harm

When one agency enforces vigorously while others do not:
- Individuals face barriers in the enforced domain
- They simultaneously face unchanged barriers in other domains
- Partial success in one domain creates false hope
- Resources expended on partial remedies are unavailable for comprehensive ones

**Implication:** Uncoordinated enforcement may be worse than no enforcement, because it consumes resources without producing outcomes.

---

## 5. What Coordination Actually Means

Coordination does not mean:
- Joint press releases
- Interagency working groups that produce reports
- Memoranda of understanding that create communication channels
- Voluntary frameworks that request industry participation

Coordination means:

### Shared Data-Governance Mandates

A single adverse event (disputed debt, criminal charge, healthcare flag) currently triggers independent data flows to independent systems governed by independent agencies.

**Required:** A unified data-governance standard that applies across all algorithmic decision systems, regardless of which agency has nominal jurisdiction. When an error is identified in one system, correction must propagate to all connected systems automatically.

**Mechanism:** Shared regulatory authority over data brokers and aggregators, with mandatory correction propagation timelines and penalties for non-compliance.

### Cross-Agency Audit Triggers

Currently, each agency audits within its domain. A healthcare algorithm audit does not trigger review of connected credit-scoring or employment-screening systems.

**Required:** Audit findings in one domain must automatically trigger review in connected domains. When HHS identifies bias in a healthcare algorithm, this must trigger CFPB review of credit impacts and EEOC review of employment impacts.

**Mechanism:** Formal audit-sharing protocols with mandatory response timelines and joint enforcement authority for cross-domain violations.

### System-Level Success Metrics (Population-Based, Not Model-Based)

Currently, success is measured by:
- Number of audits conducted
- Number of enforcement actions taken
- Percentage of models meeting fairness thresholds

These metrics can all improve while population-level harm increases.

**Required:** Success must be measured by population-level outcomes:
- Reduction in error persistence across systems
- Reduction in time-to-resolution for algorithmic disputes
- Reduction in cross-domain harm propagation

**Mechanism:** Shared data infrastructure to track individuals across domains (with appropriate privacy protections), enabling measurement of actual outcomes rather than process compliance.

---

## 6. What Happens If This Is Ignored

### Escalating Litigation

Class action litigation targeting algorithmic systems is increasing. Current cases focus on individual companies or algorithms. As plaintiffs' attorneys recognize the systemic nature of harm, litigation will increasingly target the regulatory framework itself.

**Projection:** Within 5 years, expect litigation arguing that fragmented oversight constitutes deliberate indifference to systemic discrimination. Regulatory agencies may face liability for failing to coordinate.

### Automated Discrimination Becoming Legally Untraceable

As algorithmic systems become more complex and interconnected, the causal chain from input to harm becomes longer and more diffuse. Each step in the chain can claim it is not responsible:
- The data provider didn't make the decision
- The algorithm developer didn't choose the data
- The institution using the algorithm didn't build it
- The regulator overseeing the institution doesn't oversee the algorithm

**Projection:** Within 10 years, algorithmic discrimination will be effectively impossible to prove in court, because no single entity will be demonstrably responsible. Harm will continue, but legal remedy will disappear.

### Loss of Public Trust in Health AI

Healthcare is the highest-stakes domain for algorithmic decision-making. Patients cannot opt out of algorithmic systems that determine their access to care.

**Current state:** The documented Optum algorithm affected care for millions of Black patients. Similar systems operate in Medicare Advantage, prior authorization, and care management.

**Projection:** A high-profile algorithmic failure resulting in patient deaths will occur. Public response will not distinguish between responsible and irresponsible AI deployment. Trust in all health AI will collapse, including beneficial applications.

---

## Technical Appendix: Data Sources

All findings derive from peer-reviewed analysis with complete reproducibility:

**Repository:** https://github.com/Nyx-Dynamics/algorithmic-bias-epidemiology-academic

**Barrier parameters estimated from:**
- CFPB Consumer Credit Reports Study (2022)
- FCRA dispute resolution data
- EEOC Enforcement Guidance on Arrest and Conviction Records (2012)
- Legal Services Corporation Justice Gap Report (2022)
- Obermeyer et al., Science (2019) - Healthcare algorithm bias

**Sensitivity validation:**
- Sobol global sensitivity indices (n=1,024 base samples)
- Morris elementary effects screening (r=20 trajectories)
- Bootstrap confidence intervals (n=1,000 samples)
- Signal-to-noise ratio analysis (1-30% parameter noise)

**Key robustness finding:** All conclusions demonstrated 100% robustness across bootstrap samples. The three-way interaction exceeded 70% threshold in every sample (mean: 99.6%).

---

## Contact

**AC Demidont, DO**
Nyx Dynamics LLC
acdemidont@nyxdynamics.org

---

*This document contains no recommendations that require new statutory authority. All proposed mechanisms can be implemented through existing regulatory coordination frameworks and rulemaking authority.*
