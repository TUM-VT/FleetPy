from abc import ABC
import numpy as np


UNIFORM = "uniform"
TRIANGULAR = "triangular"
POISSON = "poisson"


class RandomDistribution(ABC):
    def sample(self):
        pass


class UniformDistribution(RandomDistribution):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self, rng):
        return rng.uniform(self.low, self.high)


class TriangularDistribution(RandomDistribution):
    def __init__(self, low, mode, high):
        self.low = low
        self.mode = mode
        self.high = high

    def sample(self, rng):
        return rng.triangular(self.low, self.mode, self.high)


class PoissonDistribution(RandomDistribution):
    def __init__(self, lam):
        self.lam = lam

    def sample(self, rng):
        return rng.poisson(self.lam)


class UniformLocationDistribution(RandomDistribution):
    def __init__(self, node_ids):
        self.node_ids = np.asarray(node_ids, dtype=int)
        if len(self.node_ids) == 0:
            raise ValueError("Location distribution requires at least one node.")

    def sample(self, rng):
        idx = rng.integers(0, len(self.node_ids))
        return int(self.node_ids[idx])


class TriangularLocationDistribution(RandomDistribution):
    def __init__(self, node_ids, mode_fraction=0.5):
        self.node_ids = np.asarray(node_ids, dtype=int)
        if len(self.node_ids) == 0:
            raise ValueError("Location distribution requires at least one node.")
        if not (0.0 <= mode_fraction <= 1.0):
            raise ValueError("mode_fraction must be between 0 and 1.")
        self.max_idx = len(self.node_ids) - 1
        self.distribution = TriangularDistribution(0, self.max_idx * mode_fraction, self.max_idx)

    def sample(self, rng):
        if self.max_idx == 0:
            return int(self.node_ids[0])
        idx = int(round(self.distribution.sample(rng)))
        return int(self.node_ids[idx])

# TODO check
# class InverseTriangularLocationDistribution(RandomDistribution):
#     def __init__(self, node_ids):
#         self.node_ids = np.asarray(node_ids, dtype=int)
#         if len(self.node_ids) == 0:
#             raise ValueError("Location distribution requires at least one node.")
#         self.max_idx = len(self.node_ids) - 1
#         self.left_distribution = TriangularDistribution(0, 0, self.max_idx)
#         self.right_distribution = TriangularDistribution(0, self.max_idx, self.max_idx)
#         # TODO handle from and to hub


    def sample(self, rng):
        if self.max_idx == 0:
            return int(self.node_ids[0])
        distribution = self.left_distribution if rng.random() < 0.5 else self.right_distribution
        idx = int(round(distribution.sample(rng)))
        return int(self.node_ids[idx])


class UniformTimeDistribution(RandomDistribution):
    def __init__(self, end_time):
        if end_time <= 0:
            raise ValueError("Time distribution requires end_time > 0.")
        self.distribution = UniformDistribution(0, end_time)

    def sample(self, rng):
        return int(self.distribution.sample(rng))


class PoissonTimeDistribution(RandomDistribution):
    def __init__(self, end_time):
        # TODO check correctness
        if end_time <= 0:
            raise ValueError("Time distribution requires end_time > 0.")
        self.end_time = end_time
        self.distribution = PoissonDistribution(end_time / 2)

    def sample(self, rng):
        return min(int(self.distribution.sample(rng)), self.end_time - 1)


LOCATION_DISTRIBUTIONS = {
    UNIFORM: UniformLocationDistribution,
    TRIANGULAR: TriangularLocationDistribution,
    # "inverse_triangular": InverseTriangularLocationDistribution,
}

TIME_DISTRIBUTIONS = {
    UNIFORM: UniformTimeDistribution,
    POISSON: PoissonTimeDistribution,
}


def get_location_distribution(spatial_dist, node_ids, **kwargs):
    distribution_cls = LOCATION_DISTRIBUTIONS.get(spatial_dist)
    if distribution_cls is None:
        raise ValueError(
            f"Invalid spatial distribution: {spatial_dist}. Must be one of {sorted(LOCATION_DISTRIBUTIONS)}.")
    return distribution_cls(node_ids, **kwargs)


def get_time_distribution(temporal_dist, end_time):
    distribution_cls = TIME_DISTRIBUTIONS.get(temporal_dist)
    if distribution_cls is None:
        raise ValueError(
            f"Invalid temporal distribution: {temporal_dist}. Must be one of {sorted(TIME_DISTRIBUTIONS)}.")
    return distribution_cls(end_time)
