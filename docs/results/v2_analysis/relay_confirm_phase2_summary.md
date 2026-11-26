# Phase 2 Pre-Registered Confirmation — Frozen Analysis Summary

- Runs analyzed: **384** across **16** cells (fresh seeds `1048..1071`).
- Unified behavioral criterion: **`B_T` = (`loss_rep < 0.3` AND `deep_decoy_over_clean >= 0.9`)**.

## Comparisons (Paired on Identical Fresh Seeds)

| ID | X vs Y | n | B_T (kx vs ky) | R (rx vs ry) | U (ux vs uy) | Verdict | McNemar p |
|---|---|---:|---|---|---|---:|---:|
| PM1a_both05_vs_base | M1-both05 vs M1-base | 24 | 23 vs 0 | 1 vs 24 | 0 vs 0 | **flip** | 0.0 |
| PM1b_both025_vs_base | M1-both025 vs M1-base | 24 | 22 vs 0 | 0 vs 24 | 0 vs 0 | **flip** | 0.0 |
| PM1c_both05_vs_attn05 | M1-both05 vs M1-attn05 | 24 | 23 vs 11 | 1 vs 11 | 0 vs 1 | **shift** | 0.000488 |
| PM1d_both05_vs_mlp05 | M1-both05 vs M1-mlp05 | 24 | 23 vs 0 | 1 vs 24 | 0 vs 0 | **flip** | 0.0 |
| PM2a_lr5e4_both025_vs_base | M2a-lr5e4-both025 vs M2a-lr5e4-base | 24 | 22 vs 0 | 0 vs 24 | 0 vs 0 | **flip** | 0.0 |
| EM2b_d128_both025_vs_base | M2b-d128-both025 vs M2b-d128-base | 24 | 24 vs 0 | 0 vs 24 | 0 vs 0 | **flip** | 0.0 |
| PR1_rope_mlp_2e3_all | R1-mlp-2e3-both025 vs R1-mlp-2e3-base | 24 | 23 vs 14 | 0 vs 9 | 0 vs 0 | **shift** | 0.011719 |
| PR1_rope_mlp_2e3_h2 | R1-mlp-2e3-both025 vs R1-mlp-2e3-base | 12 | 12 vs 4 | 0 vs 8 | 0 vs 0 | **flip** | 0.007812 |
| PD1_4L_rope_vs_abs_all | D1-rope-24k vs D1-abs-24k | 24 | 24 vs 14 | 0 vs 7 | 0 vs 0 | **shift** | 0.001953 |
| PD1_4L_rope_vs_abs_h2 | D1-rope-24k vs D1-abs-24k | 12 | 12 vs 6 | 0 vs 6 | 0 vs 0 | **shift** | 0.03125 |
| ED1_4L_abs_l01_vs_base | D1-abs-l01_05-24k vs D1-abs-24k | 24 | 15 vs 14 | 7 vs 7 | 2 vs 0 | **unconfirmed** | 1.0 |

## Secondary & Blind Checks

- `secondary`: `{"SM1_attn05_le12": true, "SM1_mlp05_le8": true, "SR1a_rope_attn_1e3": true, "ER1b_rope_attn_2e3_BT": 17, "ER1b_rope_attn_2e3_R": 3}`

## All 16 Cells

| Cell | Role | L | Pos | d | lr | Steps | n | U | R | B_T | S | M | h2 B_T | h4 B_T | med loss | med R_1 | med m0 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| M1-base | primary | 2 | abs | 64 | 0.001 | 12000 | 24 | 0 | 24 | **0** | 0 | 0 | 0/12 | 0/12 | 1.0785 | 0.8256 | 0.0563 |
| M1-attn05 | primary | 2 | abs | 64 | 0.001 | 12000 | 24 | 1 | 11 | **11** | 0 | 1 | 6/12 | 5/12 | 1.071 | 0.3597 | 0.046 |
| M1-mlp05 | primary | 2 | abs | 64 | 0.001 | 12000 | 24 | 0 | 24 | **0** | 0 | 0 | 0/12 | 0/12 | 0.9905 | 0.811 | 0.0267 |
| M1-both05 | primary | 2 | abs | 64 | 0.001 | 12000 | 24 | 0 | 1 | **23** | 0 | 0 | 11/12 | 12/12 | 0.1795 | 0.4006 | 0.0248 |
| M1-both025 | primary | 2 | abs | 64 | 0.001 | 12000 | 24 | 0 | 0 | **22** | 2 | 0 | 10/12 | 12/12 | 0.1869 | 0.2068 | 0.0122 |
| M2a-lr5e4-base | primary | 2 | abs | 64 | 0.0005 | 12000 | 24 | 0 | 24 | **0** | 0 | 0 | 0/12 | 0/12 | 1.2229 | 1.5773 | 0.0745 |
| M2a-lr5e4-both025 | primary | 2 | abs | 64 | 0.0005 | 12000 | 24 | 0 | 0 | **22** | 2 | 0 | 10/12 | 12/12 | 0.1886 | 0.3442 | 0.0161 |
| M2b-d128-base | exploratory | 2 | abs | 128 | 0.0005 | 12000 | 24 | 0 | 24 | **0** | 0 | 0 | 0/12 | 0/12 | 0.7724 | 1.5316 | 0.1233 |
| M2b-d128-both025 | exploratory | 2 | abs | 128 | 0.0005 | 12000 | 24 | 0 | 0 | **24** | 0 | 0 | 12/12 | 12/12 | 0.1568 | 0.5806 | 0.0273 |
| R1-attn-1e3 | secondary | 2 | RoPE | 64 | 0.001 | 12000 | 24 | 0 | 0 | **24** | 0 | 0 | 12/12 | 12/12 | 0.1984 | 1.3063 | 0.0 |
| R1-attn-2e3 | exploratory | 2 | RoPE | 64 | 0.002 | 12000 | 24 | 0 | 3 | **17** | 0 | 4 | 9/12 | 8/12 | 0.2496 | 1.7087 | 0.0 |
| R1-mlp-2e3-base | primary | 2 | RoPE | 64 | 0.002 | 12000 | 24 | 0 | 9 | **14** | 0 | 1 | 4/12 | 10/12 | 0.2165 | 0.8084 | 0.0565 |
| R1-mlp-2e3-both025 | primary | 2 | RoPE | 64 | 0.002 | 12000 | 24 | 0 | 0 | **23** | 0 | 1 | 12/12 | 11/12 | 0.2025 | 0.2096 | 0.0119 |
| D1-abs-24k | primary | 4 | abs | 64 | 0.001 | 24000 | 24 | 0 | 7 | **14** | 2 | 1 | 6/12 | 8/12 | 0.2572 | 0.9649 | 0.0569 |
| D1-rope-24k | primary | 4 | RoPE | 64 | 0.001 | 24000 | 24 | 0 | 0 | **24** | 0 | 0 | 12/12 | 12/12 | 0.1218 | 0.9238 | 0.0568 |
| D1-abs-l01_05-24k | exploratory | 4 | abs | 64 | 0.001 | 24000 | 24 | 2 | 7 | **15** | 0 | 0 | 8/12 | 7/12 | 0.2188 | 0.2571 | 0.0197 |
