# Parameter Derivation Audit Trail

**Manuscript:** PDIG-D-26-00342 — *Structural limits of single-barrier reform in algorithmic recourse: a formal series-system model with implications for digital health*

This file documents the derivation of every stage-specific pass probability `p_i`. Each `p_i` is a **provisional, transport-limited empirical calibration** (epistemic Level 3), not a measured property of any real recourse system. Numerical model outputs derived from these values (baseline 0.0018%, maximum single-barrier gain 0.0054%, 87.6% three-way interaction share) are **model properties**, not empirical measurements.

| Column | Meaning |
|---|---|
| Pass probability | Implemented `p_i` (probability of clearing the stage) |
| Plausibility range | Prespecified sensitivity bounds |

---

## Rapid Data Transmission

- **Barrier key:** `rapid_transmission`
- **Layer:** Data Integration
- **Target construct:** Probability of correcting adverse data before it propagates to CRAs
- **Source domain:** Consumer-finance reporting practice
- **Source:** CFPB 2022 (Consumer Response Annual Report / furnisher practices)
- **Exact source statistic:** Furnishers transmit information to CRAs within ~1-2 days
- **Source page/table/figure:** CFPB 2022, furnisher-timeliness discussion
- **Mapping rule:** Short correction window before propagation -> ~30% chance of pre-propagation fix
- **Pass probability (p_i):** 0.30
- **Plausibility range:** 0.20-0.40
- **Rationale:** Rapid transmission leaves a narrow window; set near the low end of navigability.
- **Transportability limitation:** Finance-reporting timelines; not directly observed for healthcare/administrative recourse.

## Multi-System Integration

- **Barrier key:** `multi_system_integration`
- **Layer:** Data Integration
- **Target construct:** Probability of clearing errors replicated across multiple systems
- **Source domain:** Consumer-reporting accuracy
- **Source:** FTC 2013 (Report to Congress under FACT Act, credit-report accuracy study)
- **Exact source statistic:** ~20% of consumers had an error on >=1 of 3 reports
- **Source page/table/figure:** FTC 2013 accuracy study, executive summary
- **Mapping rule:** Complement of cross-system error prevalence, adjusted for multi-database copies -> ~0.55
- **Pass probability (p_i):** 0.55
- **Plausibility range:** 0.40-0.65
- **Rationale:** Most consumers are not blocked by cross-system replication for this stage.
- **Transportability limitation:** Credit-report ecosystem; generalization to 20+ healthcare databases is an assumption.

## Permanent Storage

- **Barrier key:** `permanent_storage`
- **Layer:** Data Integration
- **Target construct:** Probability of encountering data that has not yet auto-expired
- **Source domain:** Records-retention law/practice
- **Source:** CFPB 2022; FCRA Section 605 (7-year retention)
- **Exact source statistic:** 7-year retention; <20% of adverse items removed within 4 years
- **Source page/table/figure:** FCRA 15 U.S.C. 1681c (Section 605); CFPB 2022 retention discussion
- **Mapping rule:** Fraction encountering unexpired adverse data over recourse horizon -> ~0.45
- **Pass probability (p_i):** 0.45
- **Plausibility range:** 0.35-0.55
- **Rationale:** Long statutory retention makes unexpired adverse data common but not universal.
- **Transportability limitation:** Statutory retention windows differ across administrative/healthcare data domains.

## Error Detection Difficulty

- **Barrier key:** `error_detection`
- **Layer:** Data Accuracy
- **Target construct:** Probability of successfully identifying the specific material error
- **Source domain:** Consumer-reporting accuracy
- **Source:** FTC 2013 (FACT Act accuracy study)
- **Exact source statistic:** 26% of consumers identified a material error in their report
- **Source page/table/figure:** FTC 2013 accuracy study, material-error finding
- **Mapping rule:** Observed material-error identification rate, rounded -> ~0.35 for actionable detection
- **Pass probability (p_i):** 0.35
- **Plausibility range:** 0.25-0.45
- **Rationale:** Detecting the responsible error is difficult; set modestly above the raw 26%.
- **Transportability limitation:** Credit context; error visibility in algorithmic/administrative decisions may differ.

## Correction Process Barriers

- **Barrier key:** `correction_process`
- **Layer:** Data Accuracy
- **Target construct:** Probability of achieving a complete correction once an error is identified
- **Source domain:** Consumer-reporting dispute outcomes
- **Source:** FTC 2013; CFPB 2022 (dispute resolution)
- **Exact source statistic:** ~37% of disputes fully resolved in the consumer's favor
- **Source page/table/figure:** FTC 2015 follow-up / CFPB 2022 dispute-outcome data
- **Mapping rule:** Full-resolution rate, rounded -> ~0.35 achieve complete correction
- **Pass probability (p_i):** 0.35
- **Plausibility range:** 0.25-0.45
- **Rationale:** Correction procedures are complex and slow; complete resolution is a minority outcome.
- **Transportability limitation:** Dispute mechanics are finance-specific; recourse procedures elsewhere may differ.

## Incomplete Correction Propagation

- **Barrier key:** `incomplete_propagation`
- **Layer:** Data Accuracy
- **Target construct:** Probability that a correction propagates sufficiently across systems
- **Source domain:** Consumer-reporting data flows
- **Source:** CFPB 2022; FTC 2015
- **Exact source statistic:** Corrections do not auto-propagate across all downstream systems
- **Source page/table/figure:** CFPB 2022 propagation discussion
- **Mapping rule:** Estimated adequate-propagation fraction -> ~0.40
- **Pass probability (p_i):** 0.40
- **Plausibility range:** 0.30-0.50
- **Rationale:** Corrections often fail to reach all copies; set below the midpoint.
- **Transportability limitation:** Propagation topology is domain-specific; healthcare data-sharing differs.

## Awareness Gap

- **Barrier key:** `awareness_gap`
- **Layer:** Institutional
- **Target construct:** Probability an individual becomes aware that recourse is possible
- **Source domain:** Access-to-justice / legal-needs survey
- **Source:** LSC 2022 (Justice Gap Report)
- **Exact source statistic:** 92% of low-income civil legal problems get inadequate help; ~25% led to any action
- **Source page/table/figure:** LSC 2022 Justice Gap Report, headline findings
- **Mapping rule:** Fraction reaching awareness-and-action for a recourse pathway -> ~0.30
- **Pass probability (p_i):** 0.30
- **Plausibility range:** 0.20-0.40
- **Rationale:** Large justice gap implies most never become aware; set at the low-moderate end.
- **Transportability limitation:** Civil legal-needs data; direct transfer to algorithmic recourse awareness is provisional.

## Record Access Barriers

- **Barrier key:** `record_access`
- **Layer:** Institutional
- **Target construct:** Probability of obtaining one's own underlying records
- **Source domain:** Consumer-reporting access rights
- **Source:** CFPB 2022; FCRA Section 612 (free annual reports)
- **Exact source statistic:** Statutory free annual reports exist, but practical access barriers remain
- **Source page/table/figure:** FCRA 15 U.S.C. 1681j (Section 612); CFPB 2022 access discussion
- **Mapping rule:** Most-navigable stage given statutory access rights -> ~0.55
- **Pass probability (p_i):** 0.55
- **Plausibility range:** 0.45-0.65
- **Rationale:** Legal access rights make this the most navigable barrier, though not frictionless.
- **Transportability limitation:** Statutory access rights vary by domain; healthcare record access governed differently.

## Legal Knowledge Gap

- **Barrier key:** `legal_knowledge`
- **Layer:** Institutional
- **Target construct:** Probability of knowing the specific legal rights/remedies that apply
- **Source domain:** Access-to-justice / legal-needs survey
- **Source:** LSC 2022 (Justice Gap Report)
- **Exact source statistic:** ~39% believe they can use the legal system to protect themselves
- **Source page/table/figure:** LSC 2022 Justice Gap Report, legal-confidence finding
- **Mapping rule:** Knowing algorithm-specific rights is narrower than general confidence -> ~0.25
- **Pass probability (p_i):** 0.25
- **Plausibility range:** 0.15-0.35
- **Rationale:** Specific algorithmic-recourse rights are less known than general legal confidence.
- **Transportability limitation:** General legal-confidence data; algorithm-specific rights knowledge is an extrapolation.

## Legal Resource Barriers

- **Barrier key:** `legal_resources`
- **Layer:** Institutional
- **Target construct:** Probability of accessing legal resources given knowledge
- **Source domain:** Access-to-justice / legal-needs survey
- **Source:** LSC 2022 (Justice Gap Report)
- **Exact source statistic:** 46% cite cost as a barrier; ~50% of those seeking help are turned away
- **Source page/table/figure:** LSC 2022 Justice Gap Report, cost/turn-away findings
- **Mapping rule:** Fraction with knowledge who can actually secure resources -> ~0.40
- **Pass probability (p_i):** 0.40
- **Plausibility range:** 0.30-0.50
- **Rationale:** Cost and capacity constraints block many who know their rights.
- **Transportability limitation:** Civil legal-aid capacity; not a healthcare-specific resource estimate.

## Systemic Bias in Algorithms

- **Barrier key:** `systemic_bias`
- **Layer:** Institutional
- **Target construct:** Probability the governing system lacks structural bias against the individual
- **Source domain:** Healthcare algorithmic fairness
- **Source:** Obermeyer et al. 2019 (Science)
- **Exact source statistic:** Commercial risk algorithm exhibited bias reducing Black patients' identified need
- **Source page/table/figure:** Obermeyer et al. 2019, Science 366(6464):447-453
- **Mapping rule:** Context-specific bias magnitude mapped to ~0.30 chance system is not structurally biased
- **Pass probability (p_i):** 0.30
- **Plausibility range:** 0.20-0.40
- **Rationale:** Single-context commercial-algorithm study; used as a bounded illustrative proxy only.
- **Transportability limitation:** STRONG LIMITATION: one commercial cost-proxy algorithm; NOT a population-wide estimate of healthcare algorithmic bias. See Major 3 scope caveat.

---

**Consistency check:** product of the eleven pass probabilities = 1.800934e-05 = 0.0018% (matches the frozen baseline 0.0018%).

