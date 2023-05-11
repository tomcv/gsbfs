# SPDX-FileCopyrightText: 2023-present Thomas Civeit <thomas@civeit.com>
#
# SPDX-License-Identifier: MIT
"""Usage examples of the gsbfs package."""

from gsbfs.gsbfs import gso_rank, gso_boruta_select, get_expected_hits
import numpy as np
from sklearn.datasets import make_classification


def example_rank():
    """Usage example of the gso_rank() function."""
    # create instances
    n_features = 50
    n_informative = 10
    X, y = make_classification(
        n_samples=5000,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        shuffle=False,  # preserve ordering. first columns = informative features
    )
    # shuffle instances
    p = np.random.permutation(y.size)
    X, y = X[p, :], y[p]
    # rank features
    ranked_indexes, cos_sq_max = gso_rank(X, y)
    print(f"Ranked Features (total={n_features}, informative=[0,{n_informative-1}]):")
    print(ranked_indexes)


def example_hits():
    """Usage example of the get_expected_hits() function."""
    n_trials = 20
    proba = 0.5
    pmf_max = 0.005
    rejected_hits, selected_hits = get_expected_hits(n_trials, proba, pmf_max)
    print(f"Hits to be selected (n_trials={n_trials}, proba={proba}, pmf_max={pmf_max}):")
    print(selected_hits)


def example_select():
    """Usage example of the gso_boruta_select() function."""
    # create instances
    n_features = 50
    n_informative = 10
    X, y = make_classification(
        n_samples=5000,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        shuffle=False,  # preserve ordering. first columns = informative features
    )
    # shuffle instances
    p = np.random.permutation(y.size)
    X, y = X[p, :], y[p]
    # select features
    rejected_indexes, selected_indexes, indecisive_indexes = gso_boruta_select(X, y)
    print(f"Selected Features (total={n_features}, informative=[0,{n_informative-1}]):")
    print(selected_indexes)


if __name__ == '__main__':
    print("--- example_rank() ---")
    example_rank()
    print("--- example_hits() ---")
    example_hits()
    print("--- example_select() ---")
    example_select()
