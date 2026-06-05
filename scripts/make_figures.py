#!/usr/bin/env python
"""Generate result figures from run metrics into figures/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIG = Path("figures")
FIG.mkdir(exist_ok=True)


def curves(run_name, field="success_rate"):
    out = []
    for seed_dir in sorted(Path("runs").glob(f"{run_name}/seed*")):
        steps, values = [], []
        for line in (seed_dir / "metrics.jsonl").read_text().splitlines():
            rec = json.loads(line)
            steps.append(rec["env_step"])
            values.append(rec["extrinsic"][field])
        out.append((np.array(steps), np.array(values)))
    return out


def plot_arms(arms, title, path, field="success_rate", ylabel="extrinsic success rate"):
    plt.figure(figsize=(7, 4.5))
    colors = plt.cm.tab10.colors
    for i, (label, run) in enumerate(arms):
        seeds = curves(run, field)
        for steps, values in seeds:
            plt.plot(steps / 1e6, values, color=colors[i], alpha=0.25, lw=1)
        grid = seeds[0][0]
        mean = np.mean([np.interp(grid, s, v) for s, v in seeds], axis=0)
        plt.plot(grid / 1e6, mean, color=colors[i], lw=2.2, label=label)
    plt.xlabel("environment steps (millions)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("wrote", path)


# 1) first integration: Forced Coordination, three arms
plot_arms(
    [("REAP (shaping enabled)", "first_integration_reap_forced"),
     ("vanilla MAPPO", "hardness_mappo_forced"),
     ("MAPPO + RND", "first_integration_rnd_forced")],
    "First integration: Forced Coordination (sparse), 3 seeds",
    FIG / "first_integration_forced.png",
)

# 2) invariance check: Cramped Room shaped vs unshaped MAPPO+RND
plot_arms(
    [("MAPPO+RND + hand-potential PBRS", "invariance_shaped_rnd_cramped"),
     ("MAPPO+RND (unshaped)", "probe_mappo_rnd_cramped")],
    "PBRS invariance sanity check: Cramped Room (sparse), 3 seeds",
    FIG / "invariance_cramped.png",
)

# 3) final success overview across all measured protocol runs
overview = [
    ("vanilla\nCramped", "gate_mappo_cramped"),
    ("vanilla\nForced", "hardness_mappo_forced"),
    ("vanilla\nCounter", "hardness_mappo_counter"),
    ("RND\nCramped", "probe_mappo_rnd_cramped"),
    ("RND+PBRS\nCramped", "invariance_shaped_rnd_cramped"),
    ("RND\nForced", "first_integration_rnd_forced"),
    ("REAP\nForced", "first_integration_reap_forced"),
]
labels, means, errs = [], [], []
for label, run in overview:
    finals = [v[-1] for _, v in curves(run)]
    labels.append(label)
    means.append(np.mean(finals))
    errs.append(np.std(finals))
plt.figure(figsize=(8.5, 4.2))
bars = plt.bar(labels, means, yerr=errs, capsize=4,
               color=["#888", "#888", "#888", "#2a7", "#2a7", "#2a7", "#d55"])
plt.ylabel("final extrinsic success rate")
plt.title("Final success across protocol runs (5M steps, 3 seeds, sparse reward)")
plt.grid(axis="y", alpha=0.3)
for bar, m in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width() / 2, m + 0.02, f"{m:.2f}",
             ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIG / "overview_final_success.png", dpi=150)
plt.close()
print("wrote", FIG / "overview_final_success.png")

# 4) teacher-quality story: encoding beats capacity
versions = [
    ("lossless\n6k steps\n(CPU)", 0.870, None),
    ("lossless\n40k steps", 0.969, 0.759),
    ("lossless GPU\nd256/L6 120k", 0.969, 0.514),
    ("HYBRID features\n+projection 60k", 0.0, 0.0083),
]
names = [v[0] for v in versions]
invalid = [v[1] for v in versions]
loss = [v[2] for v in versions]
fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(len(names))
ax1.bar(x - 0.18, invalid, width=0.36, color="#d55", label="invalid-state rate (exact)")
ax1.axhline(0.10, color="#d55", ls="--", lw=1, alpha=0.7)
ax1.text(0.05, 0.115, "gate 0.10", color="#d55", fontsize=8)
ax1.set_ylabel("invalid-state rate after projection")
ax1.set_xticks(x)
ax1.set_xticklabels(names, fontsize=8)
ax2 = ax1.twinx()
ax2.bar(x + 0.18, [l if l is not None else np.nan for l in loss],
        width=0.36, color="#46a", label="teacher train loss (final)")
ax2.set_ylabel("teacher training loss")
ax1.set_title("Teacher quality: capacity moved loss, the encoding moved validity")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "teacher_quality_story.png", dpi=150)
plt.close()
print("wrote", FIG / "teacher_quality_story.png")
