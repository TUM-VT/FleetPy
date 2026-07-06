from abc import ABC
import numpy as np


UNIFORM = "uniform"
HUB_TRIANGULAR = "hub_triangular"
HUB_TRIANGULAR_2D = "hub_triangular_2d"
POISSON = "poisson"


def _triangular_weights(dist, scale):
    """Linear (triangular) weights: max(0, 1 - dist/scale), reaching zero at `scale`."""
    return np.maximum(0.0, 1.0 - np.asarray(dist, dtype=float) / scale)


class RandomDistribution(ABC):
    def sample(self):
        pass


class UniformDistribution(RandomDistribution):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self, rng):
        return rng.uniform(self.low, self.high)


class UniformLocationDistribution(RandomDistribution):
    def __init__(self, node_ids):
        self.node_ids = np.asarray(node_ids, dtype=int)
        if len(self.node_ids) == 0:
            raise ValueError("Location distribution requires at least one node.")

    def sample(self, rng):
        idx = rng.integers(0, len(self.node_ids))
        return int(self.node_ids[idx])


class HubTriangularLocationDistribution(RandomDistribution): 
    """Multicentric spatial distribution: sampling probability of a node decays linearly in the
    node's distance to its NEAREST hub, weight = max(0, 1 - d/scale_m), reaching zero at scale_m (a
    hard finite catchment). With two hubs this yields two density peaks (bimodal) automatically; with
    one hub it is monocentric. Distances are computed upstream (see demand_utils) so this class stays
    free of any network/coordinate coupling."""

    def __init__(self, node_ids, nearest_hub_dist, scale_m):
        self.node_ids = np.asarray(node_ids, dtype=int)
        if len(self.node_ids) == 0:
            raise ValueError("Location distribution requires at least one node.")
        if scale_m <= 0:
            raise ValueError("HubTriangularLocationDistribution requires scale_m > 0.")
        nearest_hub_dist = np.asarray(nearest_hub_dist, dtype=float)
        if nearest_hub_dist.shape != self.node_ids.shape:
            raise ValueError("nearest_hub_dist must be aligned to node_ids.")
        weights = _triangular_weights(nearest_hub_dist, scale_m)
        total = weights.sum()
        if total <= 0:
            raise ValueError(
                "HubTriangularLocationDistribution weights sum to zero: every candidate node is "
                "beyond scale_m of a hub.")
        self.probs = weights / total

    def sample(self, rng):
        idx = rng.choice(len(self.node_ids), p=self.probs)
        return int(self.node_ids[idx])


class HubTriangular2DLocationDistribution(RandomDistribution):
    """Multicentric spatial distribution that is triangular along BOTH corridor axes: the sampling
    probability of a node is the product of a triangular ramp in its along-corridor (x) distance to
    the nearest hub, weight_x = max(0, 1 - dx/scale_x), and a triangular ramp in its cross-corridor
    (y) distance to the hub row, weight_y = max(0, 1 - dy/scale_y). This peaks at the hub and decays
    to zero both toward the far corridor end and toward the width edges (an L1 pyramid). Distances
    and scales are computed upstream (see demand_utils)."""

    def __init__(self, node_ids, axis_dist_x, scale_x, row_dist_y, scale_y):
        self.node_ids = np.asarray(node_ids, dtype=int)
        if len(self.node_ids) == 0:
            raise ValueError("Location distribution requires at least one node.")
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError("HubTriangular2DLocationDistribution requires scale_x, scale_y > 0.")
        axis_dist_x = np.asarray(axis_dist_x, dtype=float)
        row_dist_y = np.asarray(row_dist_y, dtype=float)
        if axis_dist_x.shape != self.node_ids.shape or row_dist_y.shape != self.node_ids.shape:
            raise ValueError("axis_dist_x and row_dist_y must be aligned to node_ids.")
        weights = _triangular_weights(axis_dist_x, scale_x) * _triangular_weights(row_dist_y, scale_y)
        total = weights.sum()
        if total <= 0:
            raise ValueError(
                "HubTriangular2DLocationDistribution weights sum to zero: every candidate node is "
                "beyond scale_x/scale_y of a hub.")
        self.probs = weights / total

    def sample(self, rng):
        idx = rng.choice(len(self.node_ids), p=self.probs)
        return int(self.node_ids[idx])


class PoissonTimeDistribution(RandomDistribution):
    """Homogeneous Poisson arrival process on [0, end_time): the number of arrivals is
    Poisson(expected_requests), and, conditional on that count, the arrival times are i.i.d.
    Uniform(0, end_time). (For a homogeneous Poisson process the arrival times, given the count,
    are uniform -- the Poisson-ness enters only through the random count.)"""

    def __init__(self, end_time):
        if end_time <= 0:
            raise ValueError("Time distribution requires end_time > 0.")
        self.end_time = end_time
        self.distribution = UniformDistribution(0, end_time)

    def sample_count(self, expected_requests, rng):
        return int(rng.poisson(expected_requests))

    def sample(self, rng):
        return int(self.distribution.sample(rng))


LOCATION_DISTRIBUTIONS = {
    UNIFORM: UniformLocationDistribution,
    HUB_TRIANGULAR: HubTriangularLocationDistribution,
    HUB_TRIANGULAR_2D: HubTriangular2DLocationDistribution,
}

TIME_DISTRIBUTIONS = {
    POISSON: PoissonTimeDistribution,
}


def get_location_distribution(spatial_dist, node_ids, nearest_hub_dist=None, scale_m=None,
                              row_dist=None, scale_y=None):
    """Build a spatial location distribution.

    Parameters:
    - spatial_dist: One of LOCATION_DISTRIBUTIONS keys.
    - node_ids: Candidate node IDs to sample from.
    - nearest_hub_dist, scale_m: Required for HUB_TRIANGULAR and HUB_TRIANGULAR_2D (along-corridor
      distance of each candidate node to its nearest hub, aligned to node_ids; and the triangular
      cutoff distance in meters where the density reaches zero along the corridor). Ignored otherwise.
    - row_dist, scale_y: Required only for HUB_TRIANGULAR_2D (cross-corridor distance of each node to
      the hub row, aligned to node_ids; and the triangular cutoff distance across the width).
    """
    if spatial_dist == HUB_TRIANGULAR:
        if nearest_hub_dist is None or scale_m is None:
            raise ValueError("hub_triangular requires nearest_hub_dist and scale_m.")
        return HubTriangularLocationDistribution(node_ids, nearest_hub_dist, scale_m)

    if spatial_dist == HUB_TRIANGULAR_2D:
        if nearest_hub_dist is None or scale_m is None or row_dist is None or scale_y is None:
            raise ValueError("hub_triangular_2d requires nearest_hub_dist, scale_m, row_dist, scale_y.")
        return HubTriangular2DLocationDistribution(node_ids, nearest_hub_dist, scale_m, row_dist, scale_y)

    distribution_cls = LOCATION_DISTRIBUTIONS.get(spatial_dist)
    if distribution_cls is None:
        raise ValueError(
            f"Invalid spatial distribution: {spatial_dist}. Must be one of {sorted(LOCATION_DISTRIBUTIONS)}.")
    return distribution_cls(node_ids)


def get_time_distribution(temporal_dist, end_time):
    distribution_cls = TIME_DISTRIBUTIONS.get(temporal_dist)
    if distribution_cls is None:
        raise ValueError(
            f"Invalid temporal distribution: {temporal_dist}. Must be one of {sorted(TIME_DISTRIBUTIONS)}.")
    return distribution_cls(end_time)
