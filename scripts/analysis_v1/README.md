# Sky v1 analysis scripts (DR1)

These scripts implement the pre-registered analysis of sky v1 (`docs/results/v1_preregistration.md`, including
Amendment 1). Each one writes a JSON file to `docs/results/v1_analysis/`. The report is
`docs/results/v1_results.md`.

| Script | Output | Covers |
|---|---|---|
| `parity_instrument.py` | `parity_instrument.json` | checks 1 and 2 (Amendment A) |
| `parity_scaling.py` | `parity_scaling.json` | question 4 (Amendment C) |
| `markov_instrument.py` | `markov_instrument.json` | check 3 (Amendment B), chain-induced transients |
| `markov_order.py` | `markov_order.json` | question 5 (Amendments C, D) |
| `markov_context.py` | `markov_context.json` | question 6 (Amendment E) |
| `grokking.py` | `grokking.json` | questions 7 and 8 (Amendments F, G) |
| `system_checks.py` | `system_checks.json` | checks 9 and 10 (Amendments H, I) |
| `check9_request_order.py` | `check9_request_order.json` | check 9 over the exact request order (supersedes the reconstruction in `system_checks.json`) |
| `supplementary.py` | `supplementary.json` | checks raised by the completeness review, not pre-registered |

`verification.json` records the independent verification of each script's results and the completeness review.

## Running

From the repository root, extract the DR1 archives into `work/dr1` (git-ignored), then run any script:

```bash
mkdir -p work/dr1
for f in releases/DR1/lightcurves*.tar.gz releases/DR1/ricurves*.tar.gz; do tar -xzf "$f" -C work/dr1; done
PYTHONPATH=. .venv/bin/python scripts/analysis_v1/parity_instrument.py
```

`grokking.py` also reads `releases/DR0` (the v0 objects on the v1 grid, question 8) and extracts what it needs into
`work/grokking`. Every script checks that `tycho` is imported from this checkout. Paths can be changed with
`--release`, `--extracted` (or `--dr1-extracted`) and `--out`; `parity_scaling.py` takes the extracted directory as
its only argument. Each script takes 5 to 40 s on one core. Bootstraps use fixed seeds, so reruns reproduce the
JSON exactly.
