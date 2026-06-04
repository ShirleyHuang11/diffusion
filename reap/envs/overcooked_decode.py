"""Exact decoding of the Overcooked lossless grid encoding.

Inverts ``OvercookedGridworld.lossless_state_encoding`` (agent-0 view) back
to a native ``OvercookedState``. Validity is exact by construction: a grid is
valid if and only if it decodes AND re-encoding the decoded state reproduces
the grid (urgency layer aside, which only reflects the unobserved timestep).

Layer order for the agent-0 encoding (overcooked_ai 1.1.0):
  0  player_0_loc           1  player_1_loc
  2-5  player_0_orientation_{N,S,E,W}   6-9  player_1_orientation_{N,S,E,W}
  10 pot_loc  11 counter_loc  12 onion_disp_loc  13 tomato_disp_loc
  14 dish_disp_loc  15 serve_loc
  16 onions_in_pot  17 tomatoes_in_pot  18 onions_in_soup  19 tomatoes_in_soup
  20 soup_cook_time_remaining  21 soup_done  22 dishes  23 onions  24 tomatoes
  25 urgency (time-dependent; excluded from roundtrip comparison)
"""

from __future__ import annotations

import numpy as np

L_P0, L_P1 = 0, 1
L_ORI0, L_ORI1 = 2, 6
L_POT = 10
L_ONIONS_IN_POT = 16
L_TOMATOES_IN_POT = 17
L_ONIONS_IN_SOUP = 18
L_TOMATOES_IN_SOUP = 19
L_COOK_REMAINING = 20
L_SOUP_DONE = 21
L_DISHES = 22
L_ONIONS = 23
L_TOMATOES = 24
L_URGENCY = 25

COMPARE_LAYERS = list(range(25))  # everything except urgency


def _single_position(layer: np.ndarray):
    hits = np.argwhere(layer == 1)
    if len(hits) != 1 or layer.sum() != 1:
        return None
    return tuple(int(v) for v in hits[0])


def decode_lossless(grid: np.ndarray, mdp, default_timestep: int = 0):
    """Decode an integer grid (W, H, 26) to an OvercookedState, or None."""
    from overcooked_ai_py.mdp.actions import Direction
    from overcooked_ai_py.mdp.overcooked_mdp import (
        ObjectState,
        OvercookedState,
        PlayerState,
        SoupState,
    )

    grid = np.asarray(grid)
    if grid.shape != tuple(mdp.shape) + (26,):
        return None
    if not np.all(grid == np.round(grid)):
        return None
    grid = grid.astype(int)

    # players: positions and orientations
    players = []
    positions = []
    for idx, (loc_layer, ori_base) in enumerate(((L_P0, L_ORI0), (L_P1, L_ORI1))):
        pos = _single_position(grid[..., loc_layer])
        if pos is None:
            return None
        ori_hits = [
            d for d in range(4) if grid[pos[0], pos[1], ori_base + d] == 1
        ]
        if len(ori_hits) != 1 or grid[..., ori_base : ori_base + 4].sum() != 1:
            return None
        players.append((pos, Direction.INDEX_TO_DIRECTION[ori_hits[0]]))
        positions.append(pos)
    if positions[0] == positions[1]:
        return None  # players cannot overlap

    pot_locations = {tuple(p) for p in np.argwhere(grid[..., L_POT] == 1)}
    objects: dict = {}
    held: dict[int, object] = {}

    def place(obj, pos):
        if pos == positions[0]:
            if 0 in held:
                return False
            held[0] = obj
        elif pos == positions[1]:
            if 1 in held:
                return False
            held[1] = obj
        else:
            if pos in objects:
                return False
            objects[pos] = obj
        return True

    # soups: idle (ingredients still addable) and cooking/done
    idle_cells = np.argwhere(
        (grid[..., L_ONIONS_IN_POT] > 0) | (grid[..., L_TOMATOES_IN_POT] > 0)
    )
    for cell in idle_cells:
        pos = tuple(int(v) for v in cell)
        if pos not in pot_locations:
            return None  # idle soups only exist in pots
        ingredients = ["onion"] * int(grid[pos[0], pos[1], L_ONIONS_IN_POT]) + [
            "tomato"
        ] * int(grid[pos[0], pos[1], L_TOMATOES_IN_POT])
        if not 1 <= len(ingredients) <= 3:
            return None
        soup = SoupState(
            pos, ingredients=[ObjectState(name, pos) for name in ingredients],
            cooking_tick=-1,
        )
        if not place(soup, pos):
            return None

    cooking_cells = np.argwhere(
        (grid[..., L_ONIONS_IN_SOUP] > 0) | (grid[..., L_TOMATOES_IN_SOUP] > 0)
    )
    for cell in cooking_cells:
        pos = tuple(int(v) for v in cell)
        n_onion = int(grid[pos[0], pos[1], L_ONIONS_IN_SOUP])
        n_tomato = int(grid[pos[0], pos[1], L_TOMATOES_IN_SOUP])
        remaining = int(grid[pos[0], pos[1], L_COOK_REMAINING])
        done = grid[pos[0], pos[1], L_SOUP_DONE] == 1
        if not 1 <= n_onion + n_tomato <= 3:
            return None
        ingredients = [ObjectState("onion", pos)] * n_onion + [
            ObjectState("tomato", pos)
        ] * n_tomato
        if pos in pot_locations:
            if done and remaining != 0:
                return None
            soup = SoupState(pos, ingredients=ingredients, cooking_tick=0)
            # advance the cook so cook_time - tick == remaining
            target_tick = soup.cook_time - remaining
            if target_tick < 0:
                return None
            soup._cooking_tick = target_tick
        else:
            # off-pot soups are finished (held or on counters)
            if not done or remaining != 0:
                return None
            soup = SoupState(pos, ingredients=ingredients, cooking_tick=0)
            soup._cooking_tick = soup.cook_time
        if not place(soup, pos):
            return None

    for layer, name in ((L_DISHES, "dish"), (L_ONIONS, "onion"), (L_TOMATOES, "tomato")):
        for cell in np.argwhere(grid[..., layer] > 0):
            pos = tuple(int(v) for v in cell)
            if grid[pos[0], pos[1], layer] != 1:
                return None
            from overcooked_ai_py.mdp.overcooked_mdp import ObjectState as _Obj

            if not place(_Obj(name, pos), pos):
                return None

    player_states = [
        PlayerState(pos, orientation, held.get(i))
        for i, (pos, orientation) in enumerate(players)
    ]
    state = OvercookedState(
        players=player_states, objects=objects, timestep=default_timestep
    )
    return state


def encode_state(state, mdp) -> np.ndarray:
    """Agent-0 lossless encoding of a native state (integer grid)."""
    return np.asarray(mdp.lossless_state_encoding(state)[0], dtype=int)


def roundtrip_valid(grid: np.ndarray, mdp) -> bool:
    """Exact validity: the grid decodes and re-encodes to itself."""
    state = decode_lossless(grid, mdp)
    if state is None:
        return False
    re_encoded = encode_state(state, mdp)
    grid = np.asarray(grid).astype(int)
    return bool(
        np.array_equal(re_encoded[..., COMPARE_LAYERS], grid[..., COMPARE_LAYERS])
    )
