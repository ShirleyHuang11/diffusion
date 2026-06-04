"""Hand-crafted shaping potentials, looked up by name for config plumbing.

A potential provider is a callable ``(env, joint_state, steps_remaining) ->
float`` evaluated online during rollout collection (the live env gives access
to native state where the encoded vector would be awkward to decode).
"""

from __future__ import annotations

import numpy as np


def _overcooked_progress(env, joint_state, steps_remaining) -> float:
    return env.progress_potential()


def _joint_argmax_position(env, joint_state, steps_remaining) -> float:
    # generic one-hot-position potential; used by the chain test fixture
    return float(np.argmax(joint_state))


PROVIDERS = {
    "overcooked_progress": _overcooked_progress,
    "joint_argmax_position": _joint_argmax_position,
}


def make_hand_potential(name: str):
    if name not in PROVIDERS:
        raise ValueError(
            f"unknown shaping potential {name!r}; available: {sorted(PROVIDERS)}"
        )
    return PROVIDERS[name]
