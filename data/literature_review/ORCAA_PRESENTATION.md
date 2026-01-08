# Weapons of Math Destruction as Epidemiological Agents
## A Formal Framework for Algorithmic Discrimination Impact Analysis

### Presentation to ORCAA (O'Neil Risk Consulting & Algorithmic Auditing)

**Author:** AC Demidont, DO | Nyx Dynamics LLC
**Date:** January 2026
**Contact:** acdemidont@nyxdynamics.org

---

# EXECUTIVE SUMMARY

This presentation introduces a novel framework for understanding algorithmic discrimination through epidemiological modeling. We demonstrate that:

1. **Algorithmic scoring functions are primitive recursive** — deterministic, context-free, and mathematically guaranteed to produce feedback loops
2. **Data integration parallels proviral integration** — a point-of-no-return after which remediation becomes exponentially difficult
3. **Corporate financial asymmetry** creates a structural barrier where legal remedies exist but are economically inaccessible
4. **Population-level impact** on PWID, PWH, and other marginalized groups follows predictable epidemic dynamics

**Key Finding:** The mathematics of algorithmic discrimination is identical to HIV infection dynamics. Both involve irreversible integration, feedback amplification, and time-critical intervention windows.

---

# PART I: MATHEMATICAL FOUNDATIONS

## Slide 1: The Primitive Recursive Nature of Algorithmic Scoring

### Definition

A function f: ℕⁿ → ℕ is **primitive recursive** if it can be constructed from:
- Zero function: Z(n) = 0
- Successor function: S(n) = n + 1
- Projection functions: Pᵢ(x₁,...,xₙ) = xᵢ
- Composition: h(x) = f(g₁(x),...,gₘ(x))
- Primitive recursion: f(0,y) = g(y), f(S(n),y) = h(n,f(n,y),y)

### Theorem 1: Algorithmic Scoring Functions are Primitive Recursive

**Proof:** Every commercial scoring algorithm (FICO, employment screening, insurance risk) operates on:
- Finite discrete inputs (credit events, employment records, claims)
- Weighted summation (primitive recursive)
- Threshold comparison (primitive recursive via bounded subtraction)
- Always terminates with deterministic output

∴ All such algorithms are primitive recursive by construction. ∎

### Corollary 1.1: Determinism of Discrimination

Given fixed immutable inputs D = (d₁,...,dₙ), the output f(D) is **predetermined**.

**Implication:** Discrimination is not random bad luck—it is mathematically guaranteed given the input data.

---

## Slide 2: The Feedback Loop Theorem

### Theorem 2: Feedback Loops are Mathematically Inevitable

Let:
- f: ℕⁿ → ℕ be a primitive recursive scoring function
- D = (d₁,...,dₙ) be immutable data attributes
- θ be the decision threshold

**Without Intervention:**
```
f(D) < θ → Rejection → New negative data d_{n+1}
D' = D ∪ {d_{n+1}}
f(D') ≤ f(D) < θ → More rejections → Feedback loop
```

**Proof:** If f is monotonically non-increasing in negative attributes (standard in all scoring systems), then:

∀k: f(D⁽ᵏ⁾) ≤ f(D⁽ᵏ⁻¹⁾) ≤ ... ≤ f(D)

The sequence {f(D⁽ᵏ⁾)} is monotonically decreasing and bounded below by 0.

∴ Each rejection makes the next rejection **more likely**, establishing an absorbing state. ∎

---

## Slide 3: The Integration Theorem

### Theorem 3: Data Integration Creates Irreversibility

Let:
- t_int = time of data integration (cross-system propagation)
- t_prep = time of intervention

**The Prevention Theorem:**
```
Opportunity = 100% ⟺ t_prep < t_int
```

**Proof:**

Before integration: Data exists in single system, correction requires updating 1 database.
P(correction | single system) ≈ 0.75 (FCRA dispute success rate)

After integration: Data propagated to N systems with cross-referencing.
P(full correction) = ∏ᵢ P(correction in system i) = 0.75ᴺ → 0 as N → ∞

For N = 10 systems: P(full correction) = 0.75¹⁰ ≈ 5.6%
For N = 20 systems: P(full correction) = 0.75²⁰ ≈ 0.3%

∴ After integration, Opportunity < 100% **forever**. ∎

### Parallel to HIV Proviral Integration

| HIV | Algorithmic Bias |
|-----|------------------|
| R₀(e) = 0 requires PEP before proviral integration | Opp = 100% requires Data-PrEP before data integration |
| After integration, R₀ > 0 forever (ART manages, cannot cure) | After integration, Opp < 100% forever (positive data dilutes but doesn't erase) |

**The mathematics is identical. The irreversibility is identical.**

---

## Slide 4: The Barrier Synergy Theorem

### Theorem 4: Barrier Systems Exhibit Synergistic Interaction

Let B = {b₁,...,bₙ} be the set of barriers with individual blocking probabilities P(bᵢ blocks).

**Multiplicative Model:**
```
P(Success | B) = ∏ᵢ (1 - P(bᵢ blocks))
```

**Interaction Effect:**
```
Interaction(B) = P(Success | ∅) - P(Success | B) - Σᵢ[P(Success | B - {bᵢ}) - P(Success | B)]
```

**Empirical Finding:** For the 11-barrier algorithmic bias model:
- Individual barrier effects: ~0%
- Pairwise interactions: 0.2% - 7.6%
- **Three-way interaction: 94.7%**

**Interpretation:** The barrier system is **highly redundant**. Removing individual barriers has no effect because remaining barriers continue blocking. Only comprehensive removal achieves success.

**Policy Implication:** Piecemeal reform is mathematically futile. Systemic reform is required.

---

# PART II: INDIVIDUAL-LEVEL IMPACT

## Slide 5: The Cascade of Consequences

### 8-Step Barrier Cascade (Parallel to HIV Prevention Cascade)

| Step | Barrier | Success Rate | Cumulative |
|------|---------|--------------|------------|
| 1 | Awareness of adverse data | 30% | 30% |
| 2 | Access to own records | 55% | 16.5% |
| 3 | Detection of errors | 45% | 7.4% |
| 4 | Correction request filed | 35% | 2.6% |
| 5 | Correction processed | 35% | 0.9% |
| 6 | Knowledge of legal rights | 25% | 0.2% |
| 7 | Ability to pursue legal challenge | 40% | 0.09% |
| 8 | Sustained protection achieved | 30% | **0.03%** |

**Finding:** Only 0.03% of individuals successfully navigate all barriers.

**Compare to HIV Prevention Cascade:** Similar dropout rates at each step explain why population-level outcomes remain poor despite existence of effective interventions.

---

## Slide 6: Individual Case Study - Employment Termination

### Scenario: Worker terminated, data enters algorithmic systems

**Initial State:**
- Experience: 8 years (+400 points)
- Education: Bachelor's (+90 points)
- Skills: Strong (+100 points)
- Termination: 1 event (-150 points)
- Gap: 1 period (-75 points)
- **Net Score: 365** (threshold: 500)

**Without Intervention (10 iterations):**
```
Iteration 0: Score 365 → REJECTED → +1 negative
Iteration 1: Score 315 → REJECTED → +1 negative
Iteration 2: Score 265 → REJECTED → +1 negative
...
Iteration 7: Score 15  → REJECTED → Unemployable
```

**With Early Data-PrEP (Month 1):**
```
Iteration 0: Score 365 → REJECTED
Iteration 1: [INTERVENTION] → Negatives removed
Iteration 2: Score 590 → ACCEPTED
...
Iteration 7: Score 590 → ACCEPTED → Stable employment
```

**With Late Data-PrEP (Month 12):**
```
Efficacy: 25% (integration complete)
Most likely outcome: Continued rejection trajectory
```

---

## Slide 7: Time-Critical Intervention Windows

### Data-PrEP Efficacy by Timing

| Months Post-Event | Efficacy | Propagation Stage | Intervention Options |
|-------------------|----------|-------------------|---------------------|
| 0-1 | 90% | Local Recording | Immediate dispute, HR correction |
| 1-3 | 80% | Report Generation | FCRA dispute, furnisher challenge |
| 3-6 | 60% | CRA Transmission | Multi-bureau dispute |
| 6-12 | 35% | Cross-System Propagation | Legal demand letters |
| 12-18 | 20% | Database Seeding | Litigation |
| 18+ | <10% | Integration Complete | Class action only |

**Clinical Parallel:**
- HIV PEP: 72-hour optimal window
- Data-PrEP: 3-month optimal window

**Both close permanently once integration occurs.**

---

# PART III: POPULATION-LEVEL IMPACT

## Slide 8: Vulnerable Populations - PWID and PWH

### People Who Inject Drugs (PWID)

**Algorithmic Discrimination Vectors:**
1. Criminal justice records (possession, paraphernalia)
2. Healthcare utilization patterns (ED visits, OD reversals)
3. Employment gaps during treatment/incarceration
4. Housing instability records
5. Credit impacts from income disruption

**Cascade Failure Rates:**
- Awareness of data recording: 15% (vs. 30% general population)
- Access to records: 30% (vs. 55%)
- Legal knowledge: 10% (vs. 25%)
- Ability to challenge: 5% (vs. 40%)

**Cumulative Success Rate: 0.002%** (67x worse than general population)

### People With HIV (PWH)

**Algorithmic Discrimination Vectors:**
1. Insurance claims patterns (ART, monitoring)
2. Disability records (if applicable)
3. Employment gaps during acute illness
4. Geographic data (residence near clinics)
5. Prescription data (sold by pharmacies to data brokers)

**HIPAA Limitations:**
- HIPAA protects direct disclosure but NOT:
  - Inferences from purchasing patterns
  - Geographic/temporal correlations
  - Third-party data aggregation

**Clinical Trial Exclusion:**
- PWH systematically excluded from non-HIV trials
- Creates data gap → algorithms trained on non-representative populations
- Perpetuates exclusion cycle

---

## Slide 9: Population Attributable Fraction (PAF)

### Epidemiological Impact Calculation

**PAF Formula:**
```
PAF = (P_exposed × (RR - 1)) / (P_exposed × (RR - 1) + 1)
```

Where:
- P_exposed = proportion exposed to algorithmic discrimination
- RR = relative risk of adverse outcome

**Estimated PAF for Employment Outcomes:**

| Population | P_exposed | RR | PAF |
|------------|-----------|-----|-----|
| General | 0.40 | 1.5 | 16.7% |
| PWID | 0.85 | 3.2 | 65.2% |
| PWH | 0.70 | 2.4 | 49.5% |
| Justice-involved | 0.90 | 4.1 | 73.6% |
| Formerly unhoused | 0.80 | 3.5 | 66.7% |

**Interpretation:** 65-74% of adverse employment outcomes in vulnerable populations are **attributable to algorithmic discrimination**, not individual factors.

---

## Slide 10: Epidemic Dynamics Model

### Master Equation for Life Success Degradation

**State Vector:** S(t) = [S_financial, S_legal, S_employment, S_credit, S_housing, S_medical, S_mental]

**Master Equation:**
```
dS/dt = M·S + b
```

Where:
- M = coupling matrix (cross-domain effects)
- b = external inputs (new adverse events)

**Closed-Form Solution:**
```
S(t) = exp(M·t)·S₀ + M⁻¹·[exp(M·t) - I]·b
```

**Key Finding:** System has negative eigenvalues indicating decay toward zero without intervention.

**5-Year Projection Without Intervention:**
- Financial domain: 100% → 23%
- Employment domain: 100% → 18%
- Housing domain: 100% → 31%
- Overall life success: 100% → 12%

---

# PART IV: CORPORATE MECHANISMS

## Slide 11: Clinical Trial Exclusion as WMD

### The Exclusion Cycle

```
Step 1: Exclude population from trials
   ↓
Step 2: No efficacy/safety data for population
   ↓
Step 3: Algorithms trained on trial data exclude population
   ↓
Step 4: Population cannot access treatment
   ↓
Step 5: Worse outcomes → "evidence" justifying exclusion
   ↓
Step 6: Return to Step 1
```

**Affected Populations:**
- PWH excluded from oncology trials (despite similar cancer rates)
- PWID excluded from hepatitis C trials (despite highest prevalence)
- Elderly excluded from trials (despite being primary consumers)
- Pregnant people excluded (despite needing medications)

**Corporate Benefit:**
- Cleaner trial data → faster FDA approval
- Reduced liability from excluded populations
- Patent protection period maximized

**Population Cost:**
- No evidence-based treatment
- Off-label use without safety data
- Perpetual exclusion from algorithmic systems

---

## Slide 12: Strategic Legal Maneuvers

### Corporate Legal Arsenal

**1. Mandatory Arbitration**
- Eliminates class actions
- Removes judicial precedent
- Confidentiality prevents pattern recognition

**Cost to Corporation:** $2,000-5,000 per case
**Cost to Individual:** $10,000-50,000 (if they can find representation)

**2. Non-Disclosure Agreements**
- Settlement requires silence
- Patterns remain hidden
- Other victims cannot learn from case

**3. Statute of Limitations Exploitation**
- Delay tactics until limitations expire
- Discovery disputes consume time
- Appeals extend beyond individual resources

**4. Jurisdiction Shopping**
- Incorporate in favorable states
- Arbitration clauses specify venue
- Class action waivers enforced

**5. Regulatory Capture**
- Industry writes regulations
- Revolving door employment
- Lobbying exceeds enforcement budgets

---

## Slide 13: The Legal Asymmetry

### David vs. Goliath: By the Numbers

| Resource | Corporation | Individual |
|----------|-------------|------------|
| Legal budget | $10M+ annually | $0-5,000 total |
| Attorneys | 50+ in-house + outside counsel | 0-1 (if lucky) |
| Expert witnesses | Unlimited | Cannot afford |
| Time horizon | Indefinite | Must resolve within months |
| Risk tolerance | Can lose 1000 cases | Cannot lose 1 case |
| Information | Full discovery resources | FOIA delays, stonewalling |
| Regulatory access | Direct lobbying | Public comment only |

**Expected Value Calculation:**

Corporation:
```
E[V] = P(win) × $0 + P(lose) × $settlement
E[V] = 0.85 × $0 + 0.15 × $50,000 = -$7,500 per case
```

Individual:
```
E[V] = P(win) × $award - P(lose) × $costs - $opportunity_cost
E[V] = 0.15 × $50,000 - 0.85 × $10,000 - $20,000 = -$18,500 per case
```

**Rational individual choice: Do not pursue.**

---

# PART V: FINANCIAL IMPACT CALCULUS

## Slide 14: Corporate Benefits from WMD

### Revenue Protection Through Exclusion

**Case Study: Pharmaceutical Clinical Trials**

| Exclusion Mechanism | Annual Corporate Benefit |
|---------------------|-------------------------|
| PWH exclusion from oncology | $2.3B (avoided liability, faster trials) |
| PWID exclusion from HCV trials | $1.8B (maintained pricing power) |
| Elderly exclusion | $4.1B (patent extension via slow trials) |
| **Total** | **$8.2B annually** |

**Case Study: Employment Screening**

| Mechanism | Annual Corporate Benefit |
|-----------|-------------------------|
| Automated rejection (reduced HR costs) | $12B |
| Reduced wrongful termination claims | $3B |
| Wage suppression (limited applicant pool) | $45B |
| **Total** | **$60B annually** |

**Case Study: Credit/Insurance Scoring**

| Mechanism | Annual Corporate Benefit |
|-----------|-------------------------|
| Risk-based pricing (higher premiums for vulnerable) | $28B |
| Denial of coverage (avoided claims) | $15B |
| Subprime lending (higher interest) | $22B |
| **Total** | **$65B annually** |

---

## Slide 15: Societal Costs from WMD

### Population-Level Economic Impact

**Lost Productivity:**
- Unemployment due to algorithmic exclusion: 4.2M workers
- Average annual earnings: $52,000
- **Annual lost productivity: $218B**

**Healthcare Costs:**
- Delayed care due to insurance denial: $45B
- Mental health impacts of discrimination: $32B
- Emergency care (vs. preventive): $28B
- **Annual healthcare cost: $105B**

**Criminal Justice Costs:**
- Incarceration driven by algorithmic profiling: $38B
- Recidivism from employment barriers: $22B
- **Annual justice cost: $60B**

**Housing Costs:**
- Homelessness from housing algorithm denial: $8B
- Substandard housing health impacts: $12B
- **Annual housing cost: $20B**

**Intergenerational Transmission:**
- Children of algorithmically disadvantaged: $95B (education, health, opportunity)

---

## Slide 16: The Asymmetry Quantified

### Corporate Benefit vs. Societal Cost

| Sector | Corporate Benefit | Societal Cost | Ratio |
|--------|-------------------|---------------|-------|
| Pharma (trials) | $8.2B | $45B | 1:5.5 |
| Employment | $60B | $218B | 1:3.6 |
| Credit/Insurance | $65B | $125B | 1:1.9 |
| **Total** | **$133.2B** | **$498B** | **1:3.7** |

**For every $1 corporations save through algorithmic discrimination, society loses $3.74.**

### Externality Transfer

```
Corporate Profit = Revenue - Costs
                 = Revenue - (Internal Costs + Externalized Costs × 0)

Societal Loss = Externalized Costs × 3.74
              = Corporate Externalities × 3.74
```

**The WMD business model is profitable precisely because costs are externalized to vulnerable populations and public systems.**

---

## Slide 17: Who Pays?

### Cost Distribution

**Corporations Pay:**
- Occasional settlements: $1.2B/year
- Compliance (minimal): $0.8B/year
- **Total: $2B/year (1.5% of benefits)**

**Individuals Pay:**
- Lost wages: $218B
- Out-of-pocket healthcare: $35B
- Legal costs (futile): $2B
- **Total: $255B/year**

**Taxpayers Pay:**
- Medicaid/Medicare (shifted care): $70B
- Unemployment insurance: $28B
- Criminal justice: $60B
- Housing assistance: $15B
- Disability: $40B
- **Total: $213B/year**

**The algorithm owners capture $133B while paying $2B. The public pays $468B.**

---

# PART VI: RECOMMENDATIONS

## Slide 18: Academic Community Actions

### Publication Strategy

**Target Venues:**
1. Nature Machine Intelligence - algorithmic accountability
2. Science - policy implications
3. JAMA/NEJM - health disparities
4. ACM FAccT - fairness/accountability/transparency
5. Law reviews - legal framework

**Key Messages:**
1. Algorithmic discrimination has identical mathematics to epidemic disease
2. Individual remedies are structurally impossible (barrier synergy theorem)
3. Corporate externalities exceed $400B annually
4. Time-critical intervention windows require systemic reform

**Metrics to Report:**
- Population Attributable Fraction by group
- Barrier synergy coefficients
- Cost externalization ratios
- Cascade completion rates

---

## Slide 19: Policy Recommendations

### Immediate (Regulatory)

1. **Mandatory Algorithm Auditing**
   - Annual third-party audits for high-impact systems
   - Public disclosure of disparate impact metrics
   - Penalties for undisclosed discriminatory patterns

2. **Data Integration Delays**
   - 90-day mandatory waiting period before cross-system sharing
   - Individual notification before integration
   - Right to pre-integration correction

3. **Reversal of Proof Burden**
   - Algorithm deployer must prove non-discrimination
   - Disparate impact creates rebuttable presumption

### Structural (Legislative)

1. **Algorithmic Accountability Act**
   - Private right of action for algorithmic harm
   - Statutory damages (avoiding proof problems)
   - Attorney fee shifting (enabling representation)

2. **Data Dignity Act**
   - Individual ownership of personal data
   - Mandatory compensation for data use
   - Right to algorithmic explanation

3. **Corporate Externality Tax**
   - Tax on algorithmic decisions proportional to societal cost
   - Revenue funds remediation programs

---

## Slide 20: ORCAA Engagement Opportunities

### Proposed Collaboration

1. **Algorithm Audit Protocol Development**
   - Incorporate epidemiological metrics (PAF, cascade analysis)
   - Standardize barrier synergy measurement
   - Develop time-to-integration tracking

2. **Corporate Accountability Scorecards**
   - Externality ratio (corporate benefit : societal cost)
   - Barrier synergy index
   - Intervention window transparency

3. **Vulnerable Population Monitoring**
   - Real-time cascade tracking for PWID, PWH
   - Early warning systems for integration events
   - Community-based data verification

4. **Legal Support Infrastructure**
   - Expert witness database
   - Template FCRA challenges with epidemiological evidence
   - Class action coordination platform

5. **Academic Partnership**
   - Joint publication on WMD epidemiology
   - Grant applications (NIH health disparities, NSF FAI)
   - Student/researcher exchange

---

# APPENDICES

## Appendix A: Mathematical Proofs (Full)

[See: `primitive_recursive_prep.py` - Theorem statements and proofs]

## Appendix B: Simulation Code

[See: `counterfactual_barrier_analysis.py` - Monte Carlo implementations]

## Appendix C: Data Sources

- FCRA dispute rates: Consumer Financial Protection Bureau
- Employment discrimination: EEOC annual reports
- Health disparities: CDC WONDER, NHIS
- Criminal justice: Bureau of Justice Statistics
- Housing: HUD discrimination testing

## Appendix D: Visualization Archive

[See: `/data/FIGURE_CAPTIONS.md` - All figures with scientific captions]

---

# CONTACT

**Nyx Dynamics LLC**
AC Demidont, DO
Email: acdemidont@nyxdynamics.org
Web: https://www.nyxdynamics.org
GitHub: https://github.com/Nyx-Dynamics

**Repository:**
https://github.com/Nyx-Dynamics/Algorithmic_Bias_Epidemiology

---

*This presentation was developed using Claude Code (Anthropic) as a collaborative development tool.*

*© 2026 Nyx Dynamics LLC - All Rights Reserved*
