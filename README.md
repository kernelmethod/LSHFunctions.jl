# LSHFunctions.jl

- Docs: [![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://kernelmethod.github.io/LSHFunctions.jl/stable/) [![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://kernelmethod.github.io/LSHFunctions.jl/dev/)
- Build status: [![Build Status](https://travis-ci.com/kernelmethod/LSHFunctions.jl.svg?branch=master)](https://travis-ci.com/kernelmethod/LSHFunctions.jl)
- Code coverage: [![Coverage Status](https://coveralls.io/repos/github/kernelmethod/LSHFunctions.jl/badge.svg?branch=master)](https://coveralls.io/github/kernelmethod/LSHFunctions.jl?branch=master)
[![codecov](https://codecov.io/gh/kernelmethod/LSHFunctions.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kernelmethod/LSHFunctions.jl)
- DOI to cite this code: [![DOI](https://zenodo.org/badge/197700982.svg)](https://zenodo.org/badge/latestdoi/197700982)

A Julia package for [locality-sensitive hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) to accelerate similarity search.

- [What's LSH?](#whats-lsh)
- [Installation](#installation)
- [Supported similarity functions](#supported-similarity-functions)
- [Examples](#examples)

## Installation
You can install LSHFunctions.jl from the Julia REPL with

```
pkg> add LSHFunctions
```

## What's LSH?
Traditionally, if you have a data point `x`, and want to find the most similar point(s) to `x` in your database, you would compute the similarity between `x` and all of the points in your database, and keep whichever points were the most similar. For instance, this type of approach is used by the classic [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm). However, it has two major problems:

- The time to find the most similar point(s) to `x` is linear in the number of points in your database. This can make similarity search prohibitively expensive for even moderately large datasets.
- In addition, the time complexity to compute the similarity between two datapoints is typically linear in the number of dimensions of those datapoints. If your data are high-dimensional (i.e. in the thousands to millions of dimensions), every similarity computation you perform can be fairly costly.

**Locality-sensitive hashing** (LSH) is a technique for accelerating these kinds of similarity searches. Instead of measuring how similar your query point is to every point in your database, you calculate a few hashes of the query point and only compare it against those points with which it experiences a hash collision. Locality-sensitive hash functions are randomly generated, with the fundamental property that as the similarity between `x` and `y` increases, the probability of a hash collision between `x` and `y` also increases.

## Supported similarity functions
So far, there are hash functions for the similarity functions:

- Cosine similarity (`SimHash`)
- Jaccard similarity (`MinHash`)
- L1 (Manhattan / "taxicab") distance: `L1Hash`
- L2 (Euclidean) distance: `L2Hash`
- Inner product
  - `SignALSH` (recommended)
  - `MIPSHash`
- Function-space hashes (supports L1, L2, and cosine similarity)
  - `MonteCarloHash`
  - `ChebHash`

This package still needs a lot of work, including improvement to the documentation and API. In general, if you want to draw one or more new hash functions, you can use the following syntax:

## Examples
The easiest way to start constructing new hash functions is by calling `LSHFunction` with the following syntax:

```
hashfn = LSHFunction(similarity function,
                     number of hash functions to generate;
                     [LSH family-specific keyword arguments])
```

For example, the following snippet generates 10 locality-sensitive hash functions (bundled together into a single `SimHash` struct) for cosine similarity:

```julia
julia> using LSHFunction;

julia> hashfn = LSHFunction(cossim, 10);

julia> n_hashes(hashfn)
10

julia> similarity(hashfn)
cossim
```

You can then start hashing new vectors by calling `hashfn()`:

```julia
julia> x = randn(128);

julia> x_hashes = hashfn(x);
```

For more details, [check out the LSHFunctions.jl documentation](https://kernelmethod.github.io/LSHFunctions.jl/dev/).
