#!/usr/bin/env python
"""Run the teacher pipeline (warmup -> teacher -> signals -> artifacts).

Usage:
    python scripts/teacher_pipeline.py [--seed 0] [--teacher-steps 4000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reap.seeding import seed_everything  # noqa: E402
from reap.teacher_pipeline import run_pipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher-steps", type=int, default=4000)
    parser.add_argument("--min-successes", type=int, default=25)
    parser.add_argument("--max-warmup-steps", type=int, default=120_000)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--n-anchors", type=int, default=None,
                        help="cap on unique anchors; default uses all")
    parser.add_argument("--samples-per-state", type=int, default=64)
    parser.add_argument("--feasibility-samples", type=int, default=8)
    parser.add_argument("--distill-hidden", type=int, default=256)
    parser.add_argument("--distill-epochs", type=int, default=600)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hybrid", action="store_true",
                        help="run the hybrid features-encoding pipeline")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--vanilla-run", default="runs/gate_mappo_cramped/seed0")
    parser.add_argument("--rnd-run", default="runs/probe_mappo_rnd_cramped/seed0",
                        help="pass 'none' to use a single-rung ladder")
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    if args.hybrid:
        from reap.hybrid_teacher import run_hybrid_pipeline

        summary = run_hybrid_pipeline(
            layout=args.layout,
            vanilla_run=args.vanilla_run,
            rnd_run=None if args.rnd_run.lower() == "none" else args.rnd_run,
            seed=args.seed,
            teacher_steps=args.teacher_steps,
            min_successes=args.min_successes,
            max_warmup_steps=args.max_warmup_steps,
            d_model=args.d_model,
            num_layers=args.num_layers,
            nhead=args.nhead,
            n_anchors=args.n_anchors if args.n_anchors is not None else 48,
            distill_hidden=args.distill_hidden,
            distill_epochs=args.distill_epochs,
            device=args.device,
            **({"window": args.window} if args.window else {}),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    summary = run_pipeline(
        seed=args.seed,
        teacher_steps=args.teacher_steps,
        min_successes=args.min_successes,
        max_warmup_steps=args.max_warmup_steps,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        n_anchors=args.n_anchors,
        samples_per_state=args.samples_per_state,
        feasibility_samples=args.feasibility_samples,
        distill_hidden=args.distill_hidden,
        distill_epochs=args.distill_epochs,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
