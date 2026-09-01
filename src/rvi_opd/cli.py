from __future__ import annotations

import argparse

import numpy as np

from .router import Action, calibrate_thresholds, route_state
from .signals import combine_normalised_signals, compute_raw_signals


def _demo(seed: int) -> None:
    rng = np.random.default_rng(seed)
    raw = []
    for _ in range(32):
        teacher = rng.dirichlet(np.ones(64))
        student = rng.dirichlet(np.ones(64))
        raw.append(compute_raw_signals(teacher, student, top_k=8, epistemic_token_ids=[1, 2, 3, 4]))
    states = combine_normalised_signals(
        [x.disagreement for x in raw],
        [x.coverage for x in raw],
        [x.epistemic_mass for x in raw],
    )
    thresholds = calibrate_thresholds(states)
    counts = {action.value: 0 for action in Action}
    for state in states:
        counts[route_state(state, thresholds).value] += 1
    print({"thresholds": thresholds.__dict__, "routes": counts})


def main() -> None:
    parser = argparse.ArgumentParser(description="RvI-OPD CPU utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run a deterministic synthetic routing demo")
    demo.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.command == "demo":
        _demo(args.seed)


if __name__ == "__main__":
    main()
