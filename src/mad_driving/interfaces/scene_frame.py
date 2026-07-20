"""Immutable road and occlusion context supplied by scenarios."""

from dataclasses import dataclass

from mad_driving.interfaces._validation import (
    require_finite,
    require_finite_values,
    require_non_empty,
)


@dataclass(frozen=True)
class OcclusionRegion:
    """A named boundary that can hide actors from the ego vehicle."""

    region_id: str
    boundary_points_xy_m: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        require_non_empty("region_id", self.region_id)
        boundary_points = tuple(tuple(point) for point in self.boundary_points_xy_m)
        if len(boundary_points) < 2:
            raise ValueError("boundary_points_xy_m must contain at least two points")
        for point in boundary_points:
            require_finite_values("boundary_points_xy_m", point, length=2)
        object.__setattr__(self, "boundary_points_xy_m", boundary_points)


@dataclass(frozen=True)
class RoadContext:
    """Scenario-defined road rules relevant to the current observation."""

    stop_required: bool
    distance_to_conflict_point_m: float | None
    intersection_entry_prohibited: bool

    def __post_init__(self) -> None:
        if self.distance_to_conflict_point_m is not None:
            require_finite(
                "distance_to_conflict_point_m", self.distance_to_conflict_point_m
            )
