# Exact edits for PDIG-D-26-00342R1_FTC.docx

> **Correction, 6 Oct.** Items 4 and 5 below were first issued against a supporting
> information file rebuilt from LaTeX, in which the repeated-attempt table was Table H.
> The authoritative R1 supplement was then located — it had been prepared for the revision
> and never uploaded — and its table order differs. The repeated-attempt table is
> **Table G**, and the caption list has been rewritten to the real contents. If you already
> applied the earlier versions of items 4 and 5, see "Corrections to apply" at the end.

Apply to the locked track-changes file PLOS supplied. Do not retype surrounding text —
each item below is an exact find/replace so the tracked diff stays minimal.

PLOS will only accept the file back with **locked tracked changes**, so make these in
Word with tracking on, and do not accept the changes before uploading.

---

## 1. Title page — corresponding author asterisk

PLOS: *"indicate the corresponding author on the title page with an asterisk (*) and
include the corresponding author's email address."*

**Find** (author line, directly under the title):

```
A.C. Demidont, DO
```

**Replace with:**

```
A.C. Demidont, DO*
```

Add the asterisk to the **author line only** (the first occurrence, under the title), not
to the repeat inside the "Corresponding author:" block.

---

## 2. Title page — corresponding author email

**Find:**

```
Email: acdemidont@nyxdynamics.org
```

**Replace with:**

```
Email: ac.demidont@outlook.com
```

This matches the address PLOS will typeset. It is the address that will be published and
indexed.

---

## 3. Remove embedded figures

The file still contains six embedded images (`image1.png`–`image6.png`, corresponding to
Figs 1–6). PLOS requires them removed from the manuscript file; the figures are supplied
separately as `Fig1.tif`–`Fig6.tif`, which are already correctly named in the inventory.

Delete each embedded image **but keep its caption text in place.** Figure captions remain
in the manuscript; only the images come out.

No change is needed to any in-text figure citation — all 13 already use the required
`Fig 1` … `Fig 6` form, with no instances of `Figure 1`, `figure 1`, or `Fig. 1`.

---

## 4. Supporting information section — replace the whole block

**Find** (the entire `S1 File.` paragraph):

```
S1 File. Supplementary Tables S1–S7 and Supplementary Figures S1–S3: individual barrier removal effects; Shapley value attribution (seeded); Sobol sensitivity indices (seeded); a bounded-perturbation stability summary; signal-to-noise and coefficient-of-variation analysis; a full parameter-derivation audit trail (Table S6); and alternative-topology robustness (Table S7), with figures for Shapley attribution, layer-removal effects, and alternative-topology robustness.
```

**Replace with:**

```
S1 Appendix. Supporting tables. Table A, Individual barrier removal effects; Table B, Shapley value attribution; Table C, Sobol sensitivity indices; Table D, Bootstrap robustness summary; Table E, Signal-to-noise ratio analysis; Table F, Parameter derivation audit trail (all 11 barriers); Table G, Robustness to alternative topologies: correlated barriers and repeated attempts (seed 42, n = 100,000).

S1 Fig. Shapley value attribution of barrier contributions. Shapley value decomposition assigning relative contribution of each barrier to overall system success while accounting for all possible barrier removal orderings. Barriers are coloured by layer: green = Data Integration, blue = Data Accuracy, red = Institutional. Values represent fair attribution of total achievable improvement, with the five highest-contributing barriers spanning all three layers, consistent with the cross-layer nature of the modeled dynamics.

S2 Fig. Effect of layer removal on system success probability. Only complete removal across all three layers yields substantial improvement (100%). Single-layer removal produces negligible effects; two-layer combinations yield at most 7.4% (Data Accuracy + Institutional), consistent with the model’s three-way interaction structure.

S3 Fig. Robustness to alternative topologies. Baseline success probability P (left) and three-way interaction share (right) as a function of latent Gaussian-copula correlation ρ, for the equicorrelation and within-layer block variants. The comparative conclusion (single-layer ≪ coordinated) is robust to correlation while the baseline value is correlation-sensitive. The repeated-attempt results (m = 2, 3), which attenuate interaction dominance, are reported in Table G in S1 Appendix.
```

Why the old text could not stand: it announced "Tables S1–S7", but S1 was a *Note* and the
tables ran S2–S7; it attributed the parameter-derivation audit trail to Table S6, which was
bounded-perturbation robustness; and it attributed alternative-topology robustness to Table
S7, which was signal-to-noise. Every one of those pointers was wrong against the supplement
as built.

---

## 5. Main text — repoint the broken citation

**Find:**

```
are reported in Supplementary Table S7.
```

**Replace with:**

```
are reported in Table G in S1 Appendix.
```

This is the only in-text citation of a supporting-information component in the manuscript.

It was broken against every available version of the supplement: there is no Table S7 in
the file currently on record, and in the complete supplement S7 was signal-to-noise, not
the repeated-attempt results. The repeated-attempt table had no S-number at all. Table H is
now that table.

---

## 6. File inventory in Editorial Manager

| action | file | Item / Description |
|---|---|---|
| Replace | `supp-1.pdf` → `S1_Appendix.docx` | `S1 Appendix` |
| Add | `S1_Fig.tif` | `S1 Fig` |
| Add | `S2_Fig.tif` | `S2 Fig` |
| Add | `S3_Fig.tif` | `S3 Fig` |

The three figure TIFFs are in `~/Downloads/PDIG-D-26-00342/v6_upload/figures/`. They are no
longer embedded in the appendix, so uploading them does not duplicate anything.

The replacement file is `production/S1_Appendix.pdf` in the repository, generated by
`production/build_s1_appendix.py`. Flag the replacement in Author Comments — see
`production/AUTHOR_COMMENTS.md`.

---

## Checked, no action required

- **Figure citations.** 13 instances, all in the required `Fig N` form. Zero violations.
- **Figure file names.** `Fig1.tif`–`Fig6.tif` already follow the convention.
- **Author name.** Manuscript reads `A.C. Demidont, DO`, which is the intended form. The
  mismatch PLOS flagged is on the Editorial Manager side; it is addressed in Author
  Comments rather than by editing the manuscript.


---

## Corrections to apply (only if you already made the earlier edits)

Two strings need changing from what was issued yesterday.

**Find:**

```
are reported in Table H in S1 Appendix.
```

**Replace with:**

```
are reported in Table G in S1 Appendix.
```

**Find** the supporting information paragraph beginning `S1 Appendix. Supporting text,
tables and figures.` and **replace the whole paragraph** with the version in item 4 above,
which begins `S1 Appendix. Supporting tables and figures.`

The earlier caption listed a `Text A` component and eight tables, and mislabelled almost
every entry, because it was written against a differently-ordered supplement. The real
appendix has seven tables, three figures, and a scope-of-claims paragraph that is prose
rather than a labelled component.

The caption above reproduces each component's title **verbatim** from the appendix and is
generated by `production/build_si_caption.py`, so it cannot drift from the file it
describes. Re-run that script if the appendix is ever rebuilt.
