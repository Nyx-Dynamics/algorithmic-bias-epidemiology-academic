# Legacy pre-PDIG exploratory scripts (ARCHIVED — not part of PDIG-D-26-00342)

The scripts in this directory are **pre-PDIG exploratory code**. They are **not
part of the manuscript PDIG-D-26-00342** ("Synergistic barriers to algorithmic
recourse", PLOS Digital Health) and are **unvalidated**: they are not exercised
by `make all`, `make test`, or `make verify`, and their outputs are **not** used
in any figure, table, or claim in the submitted paper.

They are retained only for provenance/history. Do not cite, run, or depend on
them for reproducibility. The canonical, validated pipeline lives in the parent
`analysis/` directory and is driven entirely by the repo-root `Makefile`.

Archived here:

- `life_success_theorem.py` — early "life-success prevention theorem" scratch analysis.
- `population_analysis.py` — early population-attributable-fraction exploration.
- `barrier_analysis.py` — superseded monolith; barrier logic now lives in
  `analysis/barrier_visualization.py`.
- `alternative_topology.py` — superseded; its logic was CLI-wrapped into the
  canonical `analysis/copula_robustness.py` (see that file's header).
- `run_all_legacy.sh` — the old bespoke shell runner. Superseded by
  `reproducibility/run_all.sh`, which now delegates to `make ... all`.
