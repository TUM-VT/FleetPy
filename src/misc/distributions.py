import numpy as np


def draw_from_distribution_dict(distribution):
    """
    This function draws a key from a set of keys, which are distributed according to the respective dictionary values.
    :param distribution: dictionary: key > probability of key
    :return: randomly drawn key
    """
    normalize = sum(distribution.values())
    choices = []
    probabilities = []
    for k,v in distribution.items():
        if v == np.inf: # TODO # recheck
            return k
        choices.append(k)
        probabilities.append(v/normalize)
    return np.random.choice(choices, p=probabilities)

