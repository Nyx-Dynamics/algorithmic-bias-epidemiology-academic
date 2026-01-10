# NEJM AI Submission Instructions

## Manuscript: Algorithmic Bias Epidemiology: A Computational Framework for Modeling Algorithmic Discrimination as an Epidemiological Process

---

## Pre-Submission Checklist

### Account Setup
- [ ] Create NEJM AI author account at https://ai.nejm.org
- [ ] Register for ORCID at https://orcid.org (if not already registered)
- [ ] Have ORCID ready: ________________

### Required Files Prepared
- [x] Manuscript (PDF or Word format)
- [x] Cover Letter
- [x] Title Page
- [x] Figures (5 main + 2 supplementary)
- [x] Figure Legends
- [x] Supplementary Data
- [x] References (BibTeX available)

---

## Step-by-Step Submission Process

### Step 1: Convert Manuscript to Word Format

NEJM AI prefers .docx format over PDF/LaTeX.

**Option A: Use Pandoc (recommended)**
```bash
cd /Users/acdmbpmax/Desktop/NEJM_AI_submission
pandoc main.tex -o Manuscript.docx --bibliography=references.bib
```

**Option B: Use online LaTeX to Word converter**
- https://www.vertopal.com/en/convert/tex-to-docx
- Upload `main.tex` and `references.bib`
- Download converted .docx

**Option C: Manual conversion**
- Open PDF in Word
- Clean up formatting
- Re-insert figures at appropriate locations

### Step 2: Prepare Figure Files

Current format: PNG (acceptable)
If TIFF requested later:
```bash
# Convert PNG to TIFF (300 dpi)
sips -s format tiff individual_barrier_effects.png --out individual_barrier_effects.tiff
sips -s format tiff stepwise_comparison.png --out stepwise_comparison.tiff
sips -s format tiff interaction_heatmap.png --out interaction_heatmap.tiff
sips -s format tiff sensitivity_analysis.png --out sensitivity_analysis.tiff
sips -s format tiff snr_robustness.png --out snr_robustness.tiff
```

### Step 3: Access NEJM AI Submission Portal

1. Navigate to https://ai.nejm.org
2. Click "Submit Manuscript" or "Author Center"
3. Log in with your credentials
4. Select "New Submission"

### Step 4: Select Article Type

- Select: **Original Article**
- Confirm word count is within limit (3,000 words max for main text)
- Current word count: ~2,800 words ✓

### Step 5: Enter Manuscript Information

**Title:**
```
Algorithmic Bias Epidemiology: A Computational Framework for Modeling Algorithmic Discrimination as an Epidemiological Process
```

**Running Head (Short Title):**
```
Algorithmic Bias Epidemiology
```

**Keywords:**
```
Algorithmic discrimination; epidemiological modeling; barrier analysis; synergistic interaction; sensitivity analysis; health disparities; healthcare AI; population attributable fraction
```

### Step 6: Enter Author Information

**Author 1 (Corresponding):**
- Name: AC Demidont
- Degree: DO
- Affiliation: Nyx Dynamics LLC, United States
- Email: acdemidont@nyxdynamics.org
- ORCID: [Enter your ORCID]
- Role: Corresponding Author

### Step 7: Upload Cover Letter

Upload: `Cover_Letter.md` (convert to PDF if required)

Or copy/paste content directly if text field provided.

### Step 8: Upload Title Page

Upload: `Title_Page.md` (convert to PDF if required)

Ensure the following are included:
- [x] Full title
- [x] All author names and affiliations
- [x] Corresponding author contact
- [x] Word counts (abstract: 248, main text: ~2,800)
- [x] Funding statement
- [x] Conflict of interest disclosure
- [x] Data availability statement
- [x] AI-assisted technology disclosure
- [x] Author contributions (CRediT)

### Step 9: Upload Manuscript

Upload: `Manuscript.docx` (or `Manuscript.pdf` if Word conversion not complete)

Verify:
- [x] Line numbers present
- [x] Double-spaced
- [x] Structured abstract (Background, Methods, Results, Conclusions)
- [x] No more than 3 heading levels
- [x] References numbered consecutively

### Step 10: Upload Figures

Upload each figure separately with descriptive filenames:

| Order | Filename | Description |
|-------|----------|-------------|
| Fig 1 | `individual_barrier_effects.png` | Individual barrier removal effects |
| Fig 2 | `stepwise_comparison.png` | Strategy comparison |
| Fig 3 | `interaction_heatmap.png` | Layer interactions |
| Fig 4 | `sensitivity_analysis.png` | Global sensitivity analysis |
| Fig 5 | `snr_robustness.png` | SNR and robustness |

### Step 11: Upload Supplementary Materials

**Supplementary Figures:**
| Order | Filename | Description |
|-------|----------|-------------|
| Fig S1 | `FigS1_shapley_attribution.png` | Shapley value attribution |
| Fig S2 | `FigS2_layer_effects.png` | Layer structure |

**Supplementary Data:**
Upload: `Supplementary_Data.md` (convert to PDF if required)

Contains Tables S1-S7:
- S1: Barrier Definitions and Parameters
- S2: Individual Barrier Removal Effects
- S3: Interaction Effects (ANOVA Decomposition)
- S4: Shapley Value Attribution
- S5: Sobol Sensitivity Indices
- S6: Bootstrap Robustness Summary
- S7: Signal-to-Noise Ratio Analysis

### Step 12: Enter Disclosures

**Funding:**
```
This work was supported by Nyx Dynamics LLC. No external funding was received.
```

**Conflicts of Interest:**
```
The corresponding author (ACD) reports prior employment with Gilead Sciences, Inc. from January 2020 through November 2024 and prior ownership of company stock, which was fully divested in December 2024. Gilead Sciences, Inc. had no role in the conception, design, analysis, interpretation, or writing of this study, and provided no funding, data, materials, or input into any aspect of the work.

The corresponding author (ACD) is the owner of Nyx Dynamics, LLC, a consulting company providing advisory and fractional leadership services in healthcare, technology, and complex systems. This research was conducted independently, released as open-source work, and was not produced as part of, or in support of, any paid consulting engagement.

No other competing interests are declared.
```

**Data Availability:**
```
All code and data are available at: https://github.com/Nyx-Dynamics/algorithmic-bias-epidemiology-academic
```

**AI-Assisted Technology Disclosure:**
```
Claude (Anthropic, Claude Opus 4.5) was used to assist with manuscript formatting, organization, and editorial suggestions. All scientific content, analysis design, data interpretation, and conclusions are solely the work of the author. The author carefully reviewed and edited all AI-produced materials.
```

### Step 13: Complete COI Form (Convey System)

NEJM uses the Convey disclosure system:
1. You will receive an email link to complete the ICMJE disclosure form
2. Complete the form for each author (sole author in this case)
3. Disclose Gilead employment history as described above
4. Disclose Nyx Dynamics ownership

### Step 14: Review and Submit

1. Review all uploaded files
2. Check that figures display correctly
3. Verify author information
4. Confirm all disclosures are complete
5. Click "Submit"

### Step 15: Confirmation

- Save the confirmation email
- Note the manuscript ID number
- Expected timeline for initial decision: 2-4 weeks

---

## Post-Submission

### If Revisions Requested
- Address reviewer comments systematically
- Prepare point-by-point response letter
- Track changes in revised manuscript
- Re-upload through portal

### If Additional Materials Requested
- Convert figures to TIFF if needed
- Provide raw data files if requested
- Supply additional supplementary analyses

---

## File Locations

All submission files are available at:
- **Desktop:** `/Users/acdmbpmax/Desktop/NEJM_AI_submission/`
- **GitHub:** https://github.com/Nyx-Dynamics/algorithmic-bias-epidemiology-academic

### File Inventory

```
NEJM_AI_submission/
├── Cover_Letter.md
├── Title_Page.md
├── Manuscript.pdf
├── main.tex (LaTeX source)
├── references.bib
├── Figure_Legends.md
├── Supplementary_Data.md
├── SUBMISSION_CHECKLIST.md
├── SUBMISSION_INSTRUCTIONS.md (this file)
├── Policy_White_Paper.md
├── individual_barrier_effects.png (Fig 1)
├── stepwise_comparison.png (Fig 2)
├── interaction_heatmap.png (Fig 3)
├── sensitivity_analysis.png (Fig 4)
├── snr_robustness.png (Fig 5)
├── FigS1_shapley_attribution.png
├── FigS2_layer_effects.png
├── layer_effects.png
└── shapley_attribution.png
```

---

## Contact

**Questions about submission:**
NEJM AI Editorial Office: https://ai.nejm.org/contact

**Questions about manuscript:**
AC Demidont, DO
acdemidont@nyxdynamics.org
