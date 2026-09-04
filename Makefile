# =============================================================================
# Makefile - PDIG-D-26-00342 major-revision reproducibility pipeline
# PLOS Digital Health | Synergistic barriers to algorithmic recourse
# One-command deterministic rebuild. Baseline and revision analyses separable;
# targets fail loudly; generated filenames are deterministic.
# =============================================================================

SHELL        := /bin/bash
.DEFAULT_GOAL := help
.ONESHELL:

PYTHON       ?= python3
VENV         ?= .venv
VENV_PY      := $(VENV)/bin/python
VENV_PIP     := $(VENV)/bin/pip
SEED         ?= 42

# Prefer venv python if present, else system PYTHON (offline gate fallback).
PY           := $(shell if [ -x "$(VENV_PY)" ]; then echo "$(VENV_PY)"; else echo "$(PYTHON)"; fi)

SCRIPTS      ?= analysis
DATA         ?= data
PROCESSED    := $(DATA)/processed
PARAM_DIR    := $(DATA)/parameter_sources
BUILD        ?= build
FIGDIR       := $(BUILD)/figures
MSFIGDIR     := manuscript/figures
REPRO        := outputs/reproducibility
FIG_FORMAT   ?= tiff
FIG_DPI      ?= 300

VIZ_SCRIPT      := $(SCRIPTS)/barrier_visualization.py
SENS_SCRIPT     := $(SCRIPTS)/sensitivity_analysis.py
COPULA_SCRIPT   := $(SCRIPTS)/copula_robustness.py
DERIV_SCRIPT    := $(SCRIPTS)/build_derivation_table.py
VERIFY_SCRIPT   := $(SCRIPTS)/verify_baseline.py
MANIFEST_SCRIPT := $(SCRIPTS)/write_manifest.py

RHO_MIN      ?= 0.0
RHO_MAX      ?= 0.5
RHO_STEP     ?= 0.05
COPULA_N     ?= 100000

.PHONY: help env check-env baseline sensitivity analysis topology copula \
        systemic_bias_sensitivity derivation derivations figures figures-tif \
        tables manuscript_inputs test verify all clean distclean \
        check-copula-script check-deriv-script

help:
	@echo ""
	@echo "PDIG-D-26-00342 major-revision reproducibility pipeline"
	@echo "  make env          create/validate environment (venv or system)"
	@echo "  make baseline     reproduce baseline + assert 0.0018% / 0.0054% / 87.6%"
	@echo "  make test         pytest regression on anchors + determinism check"
	@echo "  make verify       claims-vs-code checks + baseline_verification.json + manifest"
	@echo "  make sensitivity  seeded OAT/Sobol/Morris/SNR/bootstrap (seed=$(SEED))"
	@echo "  make copula       [Major 4] alternative-topology robustness sweep"
	@echo "  make derivation   parameter_derivation.md + build/S_deriv_table.tex"
	@echo "  make figures      regenerate figures ($(FIG_FORMAT), $(FIG_DPI) dpi)"
	@echo "  make tables       regenerate computational CSV tables"
	@echo "  make all          full pipeline (env->baseline->analyses->tables->figures->test->verify)"
	@echo "  make clean        remove generated analysis outputs"
	@echo "  make distclean    also remove venv/build artifacts"
	@echo "  Using PYTHON=$(PY) SEED=$(SEED)"
	@echo ""

check-env:
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "ERROR: $(PYTHON) not found"; exit 1; }
	@$(PYTHON) -c 'import sys; assert sys.version_info[:2] >= (3,9)' || { echo "ERROR: Python 3.9+ required"; exit 1; }

env: check-env | $(REPRO)
	@if [ -d "$(VENV)" ]; then \
	  echo "venv exists: $(VENV)"; \
	else \
	  echo "NOTE: no venv; attempting create, else system $(PYTHON)."; \
	  $(PYTHON) -m venv $(VENV) 2>/dev/null && $(VENV_PIP) install --quiet --upgrade pip 2>/dev/null && \
	  ( [ -f requirements.txt ] && $(VENV_PIP) install --quiet -r requirements.txt 2>/dev/null || \
	    $(VENV_PIP) install --quiet numpy scipy matplotlib pandas 2>/dev/null ) || \
	  echo "WARN: venv/pip unavailable or offline; using system $(PYTHON)."; \
	fi
	@$(PY) -c "import numpy,scipy,matplotlib,pandas; print('deps OK:', numpy.__version__, scipy.__version__, pandas.__version__)"
	@$(PY) $(MANIFEST_SCRIPT) --seed $(SEED) --out $(REPRO)/run_manifest.json --versions-out $(REPRO)/software_versions.txt --targets env >/dev/null
	@echo "env ready (see $(REPRO)/software_versions.txt)"

$(BUILD) $(FIGDIR) $(MSFIGDIR) $(PROCESSED) $(PARAM_DIR) $(REPRO):
	@mkdir -p $@

baseline: | $(PROCESSED) $(FIGDIR) $(REPRO)
	@test -f $(VIZ_SCRIPT) || { echo "ERROR: $(VIZ_SCRIPT) not found"; exit 1; }
	@echo ">> Baseline analysis (seed=$(SEED)) -> $(PROCESSED)"
	@$(PY) $(VIZ_SCRIPT) --seed $(SEED) --outdir $(PROCESSED)
	@echo ">> Asserting frozen headline invariants (0.0018% / 0.0054% / 87.6%)"
	@$(PY) $(VERIFY_SCRIPT) --seed $(SEED) --json $(REPRO)/baseline_verification.json

sensitivity: | $(PROCESSED) $(FIGDIR)
	@test -f $(SENS_SCRIPT) || { echo "ERROR: $(SENS_SCRIPT) not found"; exit 1; }
	@echo ">> Seeded OAT/Sobol/Morris/SNR/bootstrap (seed=$(SEED))"
	@$(PY) $(SENS_SCRIPT) --seed $(SEED) --outdir $(PROCESSED) --fig-format $(FIG_FORMAT) --dpi $(FIG_DPI)

check-copula-script:
	@test -f $(COPULA_SCRIPT) || { echo "BLOCKER: $(COPULA_SCRIPT) missing"; exit 1; }

copula: check-copula-script | $(PROCESSED) $(FIGDIR)
	@echo ">> Major-4 copula sweep: rho $(RHO_MIN)..$(RHO_MAX) step $(RHO_STEP), n=$(COPULA_N), seed=$(SEED)"
	@$(PY) $(COPULA_SCRIPT) --seed $(SEED) --rho-min $(RHO_MIN) --rho-max $(RHO_MAX) --rho-step $(RHO_STEP) --n $(COPULA_N) --outdir $(PROCESSED) --figdir $(FIGDIR) --fig-format $(FIG_FORMAT) --dpi $(FIG_DPI)
	@echo "   -> $(PROCESSED)/copula_sweep.csv ; $(FIGDIR)/FigX_copula_robustness.$(FIG_FORMAT)"

topology: copula

systemic_bias_sensitivity: copula
	@echo "NOTE: Systemic-bias reclassification metrics are in copula_sweep.csv (block variant)."

analysis: sensitivity copula

check-deriv-script:
	@test -f $(DERIV_SCRIPT) || { echo "BLOCKER: $(DERIV_SCRIPT) missing"; exit 1; }

derivation: check-deriv-script | $(PARAM_DIR) $(BUILD)
	@echo ">> Building parameter derivation audit trail (11 barriers)"
	@$(PY) $(DERIV_SCRIPT) --out-md $(PARAM_DIR)/parameter_derivation.md --out-table $(BUILD)/S_deriv_table.tex
	@echo "   -> $(PARAM_DIR)/parameter_derivation.md ; $(BUILD)/S_deriv_table.tex"

derivations: derivation

# figures: regenerate ALL manuscript figures (barrier + sensitivity + copula) as
# seeded PNGs into the canonical, tracked manuscript/figures dir so committed
# figures are never stale. Use `make figures-tif` for submission-ready TIFFs.
# figures: regenerate ALL figures (barrier + sensitivity + copula) as seeded PNGs.
# Scripts also emit CSVs, so we stage everything in build/figures and copy ONLY
# the images into the tracked manuscript/figures dir (keeps it figures-only, never stale).
figures: | $(FIGDIR) $(MSFIGDIR)
	@test -f $(VIZ_SCRIPT) || { echo "ERROR: $(VIZ_SCRIPT) not found"; exit 1; }
	@test -f $(COPULA_SCRIPT) || { echo "ERROR: $(COPULA_SCRIPT) not found"; exit 1; }
	@echo ">> Regenerating ALL figures (seed=$(SEED)) -> staging in $(FIGDIR), copying images to $(MSFIGDIR)"
	@$(PY) $(VIZ_SCRIPT)  --seed $(SEED) --outdir $(FIGDIR) --fig-format png --dpi $(FIG_DPI) >/dev/null
	@$(PY) $(SENS_SCRIPT) --seed $(SEED) --outdir $(FIGDIR) --fig-format png --dpi $(FIG_DPI) >/dev/null
	@$(PY) $(COPULA_SCRIPT) --seed $(SEED) --rho-min $(RHO_MIN) --rho-max $(RHO_MAX) --rho-step $(RHO_STEP) --n $(COPULA_N) --outdir $(PROCESSED) --figdir $(FIGDIR) --fig-format png --dpi $(FIG_DPI) >/dev/null
	@cp -f $(FIGDIR)/*.png $(MSFIGDIR)/
	@echo "   -> refreshed $(MSFIGDIR)/*.png (barrier, stepwise, layer, heatmap, shapley, sensitivity, snr, copula)"

# figures-tif: submission-ready .tif at 300 dpi in build/figures (validate via PLOS NAAS)
figures-tif: | $(FIGDIR)
	@echo ">> Regenerating ALL figures as TIFF -> $(FIGDIR) (300 dpi)"
	@$(PY) $(VIZ_SCRIPT)  --seed $(SEED) --outdir $(FIGDIR) --fig-format tiff --dpi 300 >/dev/null
	@$(PY) $(SENS_SCRIPT) --seed $(SEED) --outdir $(FIGDIR) --fig-format tiff --dpi 300 >/dev/null
	@$(PY) $(COPULA_SCRIPT) --seed $(SEED) --rho-min $(RHO_MIN) --rho-max $(RHO_MAX) --rho-step $(RHO_STEP) --n $(COPULA_N) --outdir $(PROCESSED) --figdir $(FIGDIR) --fig-format tiff --dpi 300 >/dev/null
	@echo "   -> $(FIGDIR)/*.tiff (validate via PLOS NAAS before upload)"

tables: baseline sensitivity
	@echo ">> Computational CSV tables regenerated under $(PROCESSED)/"

manuscript_inputs: tables derivation copula
	@echo ">> All generated manuscript inputs built."

test:
	@echo ">> Regression tests (deterministic anchors + determinism)"
	@$(PY) -m pytest tests/test_regression.py -q

verify: | $(REPRO)
	@echo ">> Claims-vs-code verification + output existence + manifest"
	@$(PY) $(VERIFY_SCRIPT) --seed $(SEED) --json $(REPRO)/baseline_verification.json --check-outputs $(PROCESSED)/shapley_values.csv $(PROCESSED)/interaction_effects.csv $(PROCESSED)/individual_barrier_effects.csv
	@echo ">> Determinism check: two seeded sensitivity runs must be identical"
	@rm -rf $(BUILD)/_verify_a $(BUILD)/_verify_b
	@mkdir -p $(BUILD)/_verify_a $(BUILD)/_verify_b
	@$(PY) $(SENS_SCRIPT) --seed $(SEED) --outdir $(BUILD)/_verify_a >/dev/null
	@$(PY) $(SENS_SCRIPT) --seed $(SEED) --outdir $(BUILD)/_verify_b >/dev/null
	@if diff -rq $(BUILD)/_verify_a/sobol_indices.csv $(BUILD)/_verify_b/sobol_indices.csv >/dev/null && diff -rq $(BUILD)/_verify_a/snr_analysis.csv $(BUILD)/_verify_b/snr_analysis.csv >/dev/null; then \
	  echo "   PASS: seeded outputs identical across runs"; \
	else \
	  echo "   FAIL: nondeterminism detected"; exit 1; \
	fi
	@$(PY) $(MANIFEST_SCRIPT) --seed $(SEED) --out $(REPRO)/run_manifest.json --versions-out $(REPRO)/software_versions.txt --targets verify --outputs $(PROCESSED)/shapley_values.csv $(PROCESSED)/copula_sweep.csv $(REPRO)/baseline_verification.json
	@echo "   -> $(REPRO)/baseline_verification.json ; $(REPRO)/run_manifest.json"

all: env baseline sensitivity copula derivation tables figures figures-tif test verify
	@echo ""
	@echo "=== make all complete. Artifacts under $(PROCESSED)/ $(FIGDIR)/ $(BUILD)/ $(REPRO)/ ==="

clean:
	@rm -rf $(BUILD) $(FIGDIR)
	@rm -f $(PROCESSED)/copula_sweep.csv
	@echo "cleaned generated build/figure outputs (source data preserved)"

distclean: clean
	@rm -rf $(VENV) $(REPRO) analysis/__pycache__ tests/__pycache__ .pytest_cache
	@echo "removed venv + reproducibility outputs + caches"
