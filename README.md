# Gram-Schmidt Boruta Feature Selection

The `gsbfs` package provides 2 simple functions:

- `gso_rank(X_input, y_output)` that ranks candidate features.
- `gso_boruta_select(X_input, y_output)` that selects features more relevant than random probes.

Where `X_input` is the matrix of input features and `y_ouput` is the vector of output targets.
The use of the Gram-Schmidt Orthogonalization (GSO) procedure for ranking the variables of a model
that is linear with respect to its parameters was first described by
[Chen et al. (1989)](https://www.tandfonline.com/doi/abs/10.1080/00207178908953472).
The features are automatically selected using the Boruta algorithm described by
[Kursa & Rudnicki (2010)](https://www.jstatsoft.org/article/view/v036i11).

[![PyPI - Version](https://img.shields.io/pypi/v/gsbfs.svg)](https://pypi.org/project/gsbfs)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gsbfs.svg)](https://pypi.org/project/gsbfs)

## Installation

The easiest way to get `gsbfs` is to use pip:
```console
$ pip install gsbfs
```
That will install the `gsbfs` package along with all the required dependencies.

## GSO Feature Ranking

We consider a model with P candidate features. The input matrix X has P columns (i.e. Xp input vectors)
corresponding to the P features, and N rows corresponding to the N observations. 
The vector y is an N-vector that contains the values of the output for each observation.
All Xp vectors and the y vector must be centered before proceeding with the GSO algorithm.

The first iteration of the procedure consists in finding the feature vector that "best explains" the output target,
i.e. which has the smallest angle with the output vector in the N-dimensional space of observations.
To this end, we calculate cos(Xp, y)**2, the square of cosine of the angles between the feature vector and
the target vector. The feature vector for which this quantity is largest is selected.

In order to discard what is already explained by the first selected vector, all P-1 remaining input vectors,
and the output vector, are projected onto the subspace (of dimension N-1) of the selected feature.
Then, in that subspace, the projected input vector that best explains the projected  output is selected,
and the P-2 remaining feature vectors are projected onto the subspace  of the first two ranked vectors.
The procedure is repeated until all P input vectors are ranked.

## Boruta Feature Selection

The Boruta algorithm selects features that are more relevant than random probes. The latter are created by
randomly shuffling each feature of the *existing* observations. Each Xp input vector generates a new "shadow"
input vector (i.e. random probe) that is statistically irrelevant since the values of the feature for each
observation have been randomly permuted. The resulting input matrix X has 2*P columns (and still N rows),
combining the original vectors and the shadow vectors.

All the candidate features are ranked following the GSO procedure described above.
The threshold to discard features is then defined as the highest rank recorded among the shadow features.
When the rank of an original feature is higher than this threshold (i.e. better than a random probe),
it is called a "hit".

However, let us consider 2 arbitrary vectors v1 and v2, and a random vector vr.
It is important to note that there is still a probability of 50% that cos(v1, vr)**2 > cos(v1, v2)**2,
so a "lucky" random probe could randomly obtain a better rank than an original feature.
Therefore, the selection method consists in repeating the experiment (random shadow vectors + ranking) n times,
and counting the number of "hits" for each feature. Since each independent experiment can give a
binary outcome (hit or no hit), a series of n trials follows a binomial distribution.

The statistical criteria for feature selection is then the maximum value of the probability mass function
that will be considered as predictive (right tail), or as non-predictive (left tail) since the binomial
distribution PMF is symmetrical. The fraction of the PMF that is neither predictive nor non-predictive,
is considered as indecisive. For instance, considering 20 trials and a maximum probability of 0.5%,
features having 16-20 hits will be selected, and features having 0-4 hits will be rejected.

## Usage Example

Let us create a data set consisting of 50 features, including only 10 informative features,
and 5000 observations. The features will be ranked by running:

```python
from gsbfs.gsbfs import gso_rank, gso_boruta_select
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=5000, n_features=50, n_informative=10, shuffle=False)
ranked_indexes, cos_sq_max = gso_rank(X, y)
```

which will return an array containing the ranked features indexes, see `gso_rank()` function documentation.

Or the features will be selected by running:

```python
rejected_indexes, selected_indexes, indecisive_indexes = gso_boruta_select(X, y)
```
which will return an array containing the selected features indexes, see `gso_boruta_select()` function
documentation. Since the process can take several minutes or hours to complete when X is very large,
the function provides a `verbose` option to report its completion progress.

Other usage examples are provided in [example.py](example.py).

Please note that the selection process relies on random probes, which means that running the procedure multiple
times may yield different results. Moreover, when the number of observations is significantly  greater than the
number of features (N >> P), there is a higher likelihood of selecting the informative features.

## License

`gsbfs` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
