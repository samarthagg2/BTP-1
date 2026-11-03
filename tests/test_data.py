"""Sanity tests for the data layer. Run: python3 -m pytest -q tests/"""
import numpy as np
import pytest

from src.data import curve, load_dataset, load_splits, row_of_cycle


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


@pytest.fixture(scope="module")
def sp():
    return load_splits()


def test_counts(sp):
    assert (len(sp["train"]), len(sp["primary_test"]), len(sp["secondary_test"])) == (41, 43, 40)
    assert len(sp["unseen"]) == 39


def test_disjoint(sp):
    sets = [set(sp[k]) for k in ("train", "primary_test", "secondary_test", "unseen")]
    assert sum(map(len, sets)) == len(set().union(*sets))


def test_indexing(ds, sp):
    """curve(c, f, n) must return the row whose summary cycle number is n."""
    for cid in sp["train"][:10] + sp["unseen"][:5]:
        c = ds["cells"][cid]
        for n in (2, 10, 100):
            r = row_of_cycle(c, n)
            assert c["summary"]["cycle"][r] == n
            assert np.allclose(curve(c, "Qdlin", n), c["curves"]["Qdlin"][r])


def test_labels_finite(ds, sp):
    for k in ("train", "primary_test", "secondary_test", "unseen"):
        assert all(np.isfinite(ds["cells"][c]["cycle_life"]) for c in sp[k])
