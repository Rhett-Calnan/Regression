import numpy as np


def make_data(n=200, seed=42):
    """Return floor areas (m2) and fictional prices ($)."""
    rng = np.random.default_rng(seed)
    area = rng.uniform(50, 250, size=n)
    noise = rng.normal(0, 30_000, size=n)
    price = 50_000 + 3_000 * area + noise
    return area, price

def split_data(area, price, test_fraction=0.2, seed=7):
    """Shuffle paired examples; reserve a held-out test set."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(area))
    n_test = int(len(area) * test_fraction)
    test_ids = indices[:n_test]
    train_ids = indices[n_test:]
    return (area[train_ids], area[test_ids], price[train_ids], price[test_ids])
