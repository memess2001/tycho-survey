# Post-hoc Exploratory MLP Diagnostic (4x GELU MLP, d=64, B=32, n=24/cell)

> **Exploratory post-hoc diagnostic** (not pre-registered; cached separately in `work/relay_confirm_mlp_diag/`).

| Cell | n | U | R | T | S | M | T+S | T+S+M | h2 T+S | h4 T+S | R_1 med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP-lr3e-4 | 24 | 0 | 21 | 3 | 0 | 0 | 3 | 3 | 2/12 | 1/12 | 1.1605 |
| MLP-lr3e-4-c2 | 24 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | 0/12 | 0/12 | 1.5824 |
| MLP-lr5e-4 | 24 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | 0/12 | 0/12 | 1.6317 |
| MLP-lr5e-4-c05 | 24 | 0 | 22 | 2 | 0 | 0 | 2 | 2 | 1/12 | 1/12 | 1.0262 |
| MLP-lr7e-4 | 24 | 12 | 12 | 0 | 0 | 0 | 0 | 0 | 0/12 | 0/12 | 1.9175 |
| MLP-wu-lr1e-3 | 24 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | 0/12 | 0/12 | 0.7407 |
| MLP-wu-lr1e-3-c05 | 24 | 0 | 17 | 7 | 0 | 0 | 7 | 7 | 3/12 | 4/12 | 0.3317 |

## Paired Comparisons (T+S)

- **lr3e-4_vs_lr7e-4** (`MLP-lr3e-4` vs `MLP-lr7e-4`): `3 vs 0` (**unstable**, McNemar `p=0.25`)
- **lr3e-4_vs_lr5e-4** (`MLP-lr3e-4` vs `MLP-lr5e-4`): `3 vs 0` (**unconfirmed**, McNemar `p=0.25`)
- **lever_c2_at_3e-4** (`MLP-lr3e-4` vs `MLP-lr3e-4-c2`): `3 vs 0` (**unconfirmed**, McNemar `p=0.25`)
- **lever_c05_at_5e-4** (`MLP-lr5e-4-c05` vs `MLP-lr5e-4`): `2 vs 0` (**unconfirmed**, McNemar `p=0.5`)
- **lever_c05_at_wu1e-3** (`MLP-wu-lr1e-3-c05` vs `MLP-wu-lr1e-3`): `7 vs 0` (**unconfirmed**, McNemar `p=0.015625`)
