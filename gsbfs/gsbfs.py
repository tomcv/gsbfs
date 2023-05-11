# SPDX-FileCopyrightText: 2023-present Thomas Civeit <thomas@civeit.com>
#
# SPDX-License-Identifier: MIT
"""Gram-Schmidt Boruta Feature Selection."""

import numpy as np
from scipy import stats
import datetime as dt


def gso_rank(X_input, y_output, n_features=None, verbose=False):
    """Rank features using Gram-Schmidt Orthogonalization procedure.

    Args:
        X_input (numpy.ndarray): 2D array of input features, 1 raw per instance
        y_output (numpy.ndarray): 1D array of output targets
        n_features (int): If not None, stop after n_features features are ranked
        verbose (bool): Report about calculation progress

    Returns:
        ranked_indexes (numpy.ndarray): Array of ranked features indexes
        cos_sq_max (numpy.ndarray): Array of calculated cosine squared
    """
    X, y = X_input.copy(), y_output.copy()  # array content will be modified
    original_indexes = np.arange(X.shape[1])
    n_features = n_features if n_features else X.shape[1]
    ranked_indexes = []
    cos_sq_max = []
    # X and y must be centered
    X = X - np.mean(X, axis=0)
    y = y - np.mean(y)
    if verbose:
        begin = dt.datetime.now()
        print(f"<gso_rank> projecting {n_features} features...")
    while len(ranked_indexes) != n_features:
        # normalize all vectors, ranking based on (xk, y) angle, norm does not matter
        X = X / np.linalg.norm(X, axis=0)
        y = y / np.linalg.norm(y)
        # find xk that maximizes cos2(xk, y) i.e. best explains y
        cos_sq = np.matmul(X.transpose(), y)**2
        if np.isnan(cos_sq).any():
            print(f"<gso_rank> WARNING: cos_sq has NaN values")
        k = np.argmax(cos_sq)
        xk = X[:, k]
        ranked_indexes.append(original_indexes[k])
        cos_sq_max.append(cos_sq[k])
        # remove xk from X
        X = np.delete(X, k, axis=1)
        original_indexes = np.delete(original_indexes, k)
        # project all vectors onto xk: v = v_para + v_ortho
        # replace vectors by v_ortho = v - v_para
        for i in range(X.shape[1]):
            X[:, i] = X[:, i] - (np.vdot(X[:, i], xk) * xk)
        y = y - (np.vdot(y, xk) * xk)
    if verbose:
        end = dt.datetime.now()
        delta = (end - begin).total_seconds()
        print(f"<gso_rank> projected {n_features} features in {delta:.3f} sec")
    return np.array(ranked_indexes), np.array(cos_sq_max)


def get_expected_hits(n_trials, proba, pmf_max):
    """Calculate expected number of hits based on binomial distribution parameters.

    Args:
        n_trials (int): Number of independent experiments
        proba (float): Probability of success
        pmf_max (float): Maximum value of the probability mass function

    Returns:
        rejected_hits (numpy.ndarray): Array of hits that are rejected
        selected_hits (numpy.ndarray): Array of hits that are selected
    """
    pmf = np.array([stats.binom.pmf(x, n_trials, proba) for x in range(n_trials + 1)])
    rejected_hits = np.where((pmf < pmf_max) & (np.arange(pmf.size) < (0.5 * pmf.size)))[0]
    selected_hits = np.where((pmf < pmf_max) & (np.arange(pmf.size) > (0.5 * pmf.size)))[0]
    return rejected_hits, selected_hits


def gso_boruta_select(X_input, y_output, n_trials=20, proba=0.5, pmf_max=0.005, verbose=False):
    """Select features using Boruta procedure.

    Args:
        X_input (numpy.ndarray): 2D array of input features, 1 raw per instance
        y_output (numpy.ndarray): 1D array of output targets
        n_trials (int): Number of independent experiments
        proba (float): Probability of success
        pmf_max (float): Maximum value of the probability mass function
        verbose (bool): Report about calculation progress

    Returns:
        rejected_indexes (numpy.ndarray): Array of rejected features indexes
        selected_indexes (numpy.ndarray): Array of selected features indexes
        indecisive_indexes (numpy.ndarray): Array of indecisive features indexes
    """
    n_features = X_input.shape[1]
    rejected_hits, selected_hits = get_expected_hits(n_trials, proba, pmf_max)
    all_ranked_indexes = []
    if verbose:
        begin = dt.datetime.now()
        print(f"<gso_boruta_select> processing {n_features} features...")
    for repeat in range(n_trials):
        if verbose:
            print(f"<gso_boruta_select> trial #{repeat+1}")
        # make X_shadow random probes by randomly permuting each column of X
        X_shadow = X_input.copy()
        for i in range(n_features):
            p = np.random.permutation(X_shadow.shape[0])
            X_shadow[:, i] = X_shadow[p, i]
        # combine X and X_shadow
        X_combined = np.hstack([X_input, X_shadow])
        # rank combined original+shadow features
        ranked_indexes, cos_sq_max = gso_rank(X_combined, y_output, verbose=verbose)
        all_ranked_indexes.append(ranked_indexes)
    # count which features have better ranks than the shadow features
    hits = np.zeros(n_features).astype(int)
    for ranked_indexes in all_ranked_indexes:
        best_shadow_index = np.min(np.where(ranked_indexes >= n_features)[0])
        hits_indexes = ranked_indexes[0:best_shadow_index]
        hits[hits_indexes] += 1
    # select/reject features based on number of hits
    rejected_indexes, selected_indexes, indecisive_indexes = [], [], []
    for feature_index in range(hits.size):
        if hits[feature_index] in rejected_hits:
            rejected_indexes.append(feature_index)
        elif hits[feature_index] in selected_hits:
            selected_indexes.append(feature_index)
        else:
            indecisive_indexes.append(feature_index)
    if verbose:
        end = dt.datetime.now()
        delta = (end - begin).total_seconds()
        print(f"<gso_boruta_select> processed {n_features} features in {delta:.3f} sec")
    return np.array(rejected_indexes), np.array(selected_indexes), np.array(indecisive_indexes)
