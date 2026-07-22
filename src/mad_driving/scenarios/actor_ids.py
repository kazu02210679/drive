"""Stable actor identifiers shared by simulator adapters and snapshot construction."""

from __future__ import annotations

from numbers import Integral
from typing import Any
from uuid import UUID


def stable_actor_id(object_key: object, simulator_object: Any) -> str:
    """Return the project-stable ID for a simulator actor."""

    actor_id = str(getattr(simulator_object, "name", object_key))
    try:
        UUID(actor_id)
    except ValueError:
        return actor_id
    random_seed = getattr(simulator_object, "random_seed", None)
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        return actor_id
    return f"metadrive-{type(simulator_object).__name__}-{int(random_seed)}"
