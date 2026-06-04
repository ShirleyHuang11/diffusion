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
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    summary = run_pipeline(
        seed=args.seed,
        teacher_steps=args.teacher_steps,
        min_successes=args.min_successes,
        max_warmup_steps=args.max_warmup_steps,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
