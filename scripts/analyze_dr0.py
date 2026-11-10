"""First-look analysis of a Tycho data release: positive controls, parity and grokking phase structure, RI events.

Usage: python scripts/analyze_dr0.py releases/DR0
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1] if len(sys.argv) > 1 else "releases/DR0")
o = pd.read_parquet(R / "objects.parquet")
e = pd.read_parquet(R / "ri_events.parquet")
runs = pd.read_parquet(R / "ri_runs.parquet")
cfg = pd.json_normalize(o["task_cfg_json"].map(json.loads))
cfg.index = o.index
o = pd.concat([o, cfg.add_prefix("c_")], axis=1)
e = e.merge(o[["object_id", "region", "label", "t_test_acc90", "t_train_acc99", "seed"]], on="object_id")
e["subset"] = e["subset_json"].map(lambda s: tuple(json.loads(s)))
pd.set_option("display.width", 200, "display.max_columns", 30)


def loglog_slope(x, y):
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 3:
        return float("nan"), float("nan"), int(m.sum())
    X, Y = np.log10(x[m]), np.log10(y[m])
    A = np.vstack([X, np.ones_like(X)]).T
    coef, res, *_ = np.linalg.lstsq(A, Y, rcond=None)
    yhat = A @ coef
    r2 = 1 - ((Y - yhat) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    return coef[0], r2, int(m.sum())


print("=" * 100, "\nA. POSITIVE CONTROLS")
gt = e[(e.matches_gt == True) & (e.kind == "activation")].sort_values("step").groupby(["object_id", "order"]).first().reset_index()
cp = gt[gt.region == "control-parity"].merge(o[["object_id", "c_k", "c_width"]], on="object_id")
cp["lead"] = cp["t_test_acc90"] - cp["step"]
print("control-parity: GT onset vs test-acc-90% step (lead = t_acc90 - onset)")
print(cp.groupby(["c_k", "c_width"]).agg(n=("step", "size"), onset_med=("step", "median"), onset_min=("step", "min"),
      onset_max=("step", "max"), acc90_med=("t_test_acc90", "median"), lead_med=("lead", "median"),
      lead_min=("lead", "min"), frac_lead_pos=("lead", lambda s: (s > 0).mean())).round(1))
cm = gt[gt.region == "control-markov"].merge(o[["object_id", "c_couplings"]], on="object_id")
print("\ncontrol-markov: GT onset step by coupling structure and order")
print(cm.groupby(["c_couplings", "order"]).step.describe()[["count", "min", "50%", "max"]])
for c, g in cm.groupby("c_couplings"):
    piv = g.pivot_table(index="object_id", columns="order", values="step")
    if piv.shape[1] >= 2:
        cols = sorted(piv.columns)
        for a, b in zip(cols, cols[1:]):
            ratio = (piv[b] / piv[a].clip(lower=1)).dropna()
            print(f"  {c}: onset(order {b}) / onset(order {a}) median {ratio.median():.1f}  range [{ratio.min():.1f}, {ratio.max():.1f}]  n={len(ratio)}")
ctrl = e[e.region.str.startswith("control")]
spur = ctrl[(ctrl.matches_gt != True) & (ctrl.kind == "activation")]
print("\nnon-GT activations in controls per object: median", spur.groupby("object_id").size().reindex(
    o[o.region.str.startswith('control')].object_id, fill_value=0).median(), "| their peak RI median", round(spur.peak_ri.median(), 4),
    "| GT peak RI median", round(gt[gt.region.str.startswith('control')].merge(e[['object_id','order','peak_ri','matches_gt','kind']][(e.matches_gt==True)&(e.kind=='activation')], on=['object_id','order']).peak_ri_y.median() if False else e[(e.matches_gt==True)&(e.kind=='activation')&e.region.str.startswith('control')].peak_ri.median(), 4))

print("=" * 100, "\nB. PARITY-WIDE (photometry)")
p = o[o.region == "parity-wide"].copy()
p["learned"] = p.final_test_acc >= 0.9
p["n_train_s"] = p.c_n_train.fillna(-1).astype(int).replace(-1, "online")
print("success rate (final test acc >= 0.9) by n, k:")
print(p.pivot_table(index="c_n", columns="c_k", values="learned", aggfunc="mean").round(2))
print("\nsuccess rate by data size and k:")
print(p.pivot_table(index="n_train_s", columns="c_k", values="learned", aggfunc="mean").round(2))
print("\nmedian t_test_acc90 (learned runs) by n, k:")
print(p[p.learned].pivot_table(index="c_n", columns="c_k", values="t_test_acc90", aggfunc="median").round(0))
print("\nlabel mix by k:"); print(p.pivot_table(index="label", columns="c_k", values="object_id", aggfunc="count").fillna(0).astype(int))
for k in sorted(p.c_k.unique()):
    for lr in sorted(p.train_lr.unique()):
        g = p[(p.c_k == k) & (p.learned) & (p.train_lr == lr) & (p.n_train_s == "online")]
        med = g.groupby("c_n").t_test_acc90.median()
        s, r2, n = loglog_slope(med.index.values.astype(float), med.values)
        print(f"  k={k} lr={lr}: t_acc90 ~ n^{s:.2f} (R2 {r2:.2f}, {n} n-values, {len(g)} runs)")

print("=" * 100, "\nC. GROKKING REGION (modular addition)")
gk = o[o.region == "grokking"].copy()
gk["grok"] = gk.label == "grokking"
print("grokking frequency by weight decay x train fraction:")
print(gk.pivot_table(index="train_weight_decay", columns="c_train_frac", values="grok", aggfunc="mean").round(2))
print("\nlabel mix by weight decay:"); print(gk.pivot_table(index="label", columns="train_weight_decay", values="object_id", aggfunc="count").fillna(0).astype(int))
g2 = gk[gk.grok]
print("\nmedian grok_delay (steps from train 99% to test 90%) by weight decay and lr:")
print(g2.pivot_table(index="train_weight_decay", columns="train_lr", values="grok_delay", aggfunc="median").round(0))
for p_ in sorted(g2.c_p.unique()):
    med = g2[g2.c_p == p_].groupby("train_weight_decay").t_test_acc90.median()
    s, r2, n = loglog_slope(med.index.values.astype(float), med.values)
    print(f"  p={p_}: t_test_acc90 ~ wd^{s:.2f} (R2 {r2:.2f}, {n} wd values)")

print("=" * 100, "\nD. RI FOLLOW-UPS")
fo = runs[runs.kind == "followup"].merge(o[["object_id", "region", "label"]], on="object_id")
print("follow-ups by region x label:"); print(fo.pivot_table(index="label", columns="region", values="object_id", aggfunc="count").fillna(0).astype(int))
ge = e[e.region == "grokking"]
per = ge.groupby("object_id").agg(n_act=("kind", lambda s: (s == "activation").sum()), label=("label", "first"),
                                   t_train99=("t_train_acc99", "first"), t_acc90=("t_test_acc90", "first"))
late = ge[(ge.kind == "activation") & (ge.step_lo >= ge.t_train_acc99)]
per["n_late"] = late.groupby("object_id").size().reindex(per.index, fill_value=0)
per["n_late_strong"] = late[late.peak_ri >= 0.02].groupby("object_id").size().reindex(per.index, fill_value=0)
print("\ngrokking-region follow-ups: activations / late activations per object, by label")
print(per.groupby("label").agg(n=("n_act", "size"), act_med=("n_act", "median"), late_med=("n_late", "median"),
                               frac_any_late=("n_late", lambda s: (s > 0).mean()), strong_late_med=("n_late_strong", "median"),
                               frac_any_strong_late=("n_late_strong", lambda s: (s > 0).mean())).round(2))
gg = late.merge(per[["t_acc90"]], left_on="object_id", right_index=True)
gg = gg[gg.label == "grokking"]
if len(gg):
    rel = (gg.step / gg.t_acc90_y if "t_acc90_y" in gg else gg.step / gg["t_acc90"])
    print("\nlate activations in grokking objects: onset / t_test_acc90  quantiles",
          np.round(rel.quantile([.1, .25, .5, .75, .9]).values, 2), " (n =", len(rel), ")")
print("\npeak RI of events by region (quantiles 10/50/90%):")
print(e.groupby("region").peak_ri.quantile([.1, .5, .9]).unstack().round(4))
print("\nevent order mix by region:"); print(e[e.kind == "activation"].pivot_table(index="region", columns="order", values="event_id", aggfunc="count").fillna(0).astype(int))
pw = e[e.region == "parity-wide"]
print("\nparity-wide follow-up events:"); print(pw.groupby(["object_id", "label"]).agg(n=("kind", "size"), gt=("matches_gt", "sum"), maxpeak=("peak_ri", "max")))
