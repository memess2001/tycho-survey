# Post-hoc Exploratory Optimization Test: Layer-0 Attn0 + MLP0 Residual Scaling

> **Exploratory post-hoc test** on full modern GPT recipe (`4x GELU MLP + AdamW wd=0.1 + 200-step warmup, lr=1e-3, B=32, d=64, n=24/arm`).

| Arm | Scaling (Attn0, MLP0) | n | U | R | T | S | M | T+S | h2 T+S | h4 T+S | med R_1 | med m0 | med loss | vs base (McNemar p) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MLP-wu-lr1e-3 | (1.0, 1.0) | 24 | 0 | 24 | 0 | 0 | 0 | **0** | 0/12 | 0/12 | 0.7407 | 0.0552 | 1.061 | baseline |
| MLP-wu-lr1e-3-c05 | (0.5, 1.0) | 24 | 0 | 17 | 7 | 0 | 0 | **7** | 3/12 | 4/12 | 0.3317 | 0.0421 | 1.073 | **unconfirmed** (`p=0.015625`) |
| MLP-wu-lr1e-3-m05 | (1.0, 0.5) | 24 | 0 | 23 | 1 | 0 | 0 | **1** | 0/12 | 1/12 | 0.7328 | 0.0264 | 0.994 | **unconfirmed** (`p=1.0`) |
| MLP-wu-lr1e-3-both05 | (0.5, 0.5) | 24 | 0 | 3 | 21 | 0 | 0 | **21** | 11/12 | 10/12 | 0.3903 | 0.0239 | 0.183 | **flip** (`p=1e-06`) |
| MLP-wu-lr1e-3-both025 | (0.25, 0.25) | 24 | 0 | 0 | 24 | 0 | 0 | **24** | 12/12 | 12/12 | 0.2119 | 0.0119 | 0.183 | **flip** (`p=0.0`) |
