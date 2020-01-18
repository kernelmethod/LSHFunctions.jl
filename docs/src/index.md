# LSH.jl

LSH.jl is a Julia package for performing [locality-sensitive hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) with various similarity functions.

## Introduction
One of the simplest methods for classifying, categorizing, and grouping data is to measure how similarities pairs of data points are. For instance, the classical [``k``-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) searches an input space ``X`` by taking a query point ``x\in X`` and a similarity function

```math
s:X\times X\to\mathbb{R}
```

It then computes ``s(x,y)`` for every point ``y`` in a database, and keeps the ``k`` points that are closest to ``x``.

Broadly, there are two computational issues with this approach:

- First, the database may be massive, much larger than could possibly fit in memory. This would make the brute-force approach of computing ``s(x,y)`` for every point ``y`` in the database far too expensive to be practical.
- Second, the dimensionality of the data may be such that computing ``s(x,y)`` is itself expensive. In addition, the similarity function itself may simply be intrinsically difficult to compute. For instance, calculating Wasserstein distance entails solving a very high-dimensional linear program.

## Locality-sensitive hashing
*Locality-sensitive hashing* (LSH) is a technique for accelerating similarity search that works by using a hash function on the query point ``x`` and limiting similarity search to only those points in the database that experience a hash collision with ``x``. The hash functions that are used are randomly generated from a family of *locality-sensitive hash functions*. These hash functions have the property that ``Pr[h(x) = h(y)]`` (i.e., the probability of a hash collision) increases the more similar that ``x`` and ``y`` are.

LSH.jl is a package that provides definitions of locality-sensitive hash functions for a variety of different similarities. Currently, LSH.jl supports hash functions for

- Cosine similarity (`cossim`)
- Jaccard similarity (`jaccard`)
- ``\ell^1`` (Manhattan / "taxicab") distance (`ℓ1`)
- ``\ell^2`` (Euclidean) distance (`ℓ2`)
- Inner product (`inner_prod`)
- Function-space hashes (`L1`, `L2`, and `cossim`)

## Contents

```@contents
```
