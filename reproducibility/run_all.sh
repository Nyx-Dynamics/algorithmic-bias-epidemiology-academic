#!/bin/bash
# Canonical reproducibility entrypoint for PDIG-D-26-00342.
# The Makefile at the repo root is the single source of truth for the pipeline
# (env -> baseline -> analyses -> tables -> figures -> test -> verify).
# This wrapper just delegates to it. The old bespoke runner is archived at
# analysis/archive/legacy_pre_pdig/run_all_legacy.sh (unvalidated, pre-PDIG).
set -euo pipefail
exec make -C "$(cd "$(dirname "$0")/.." && pwd)" all
