import random
import numpy as np


def set_random_seed(seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(get_random_seed())


def get_random_seed():
    return np.random.randint(2 ** 32)