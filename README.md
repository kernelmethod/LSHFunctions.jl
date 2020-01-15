# LSH.jl

[![Build Status](https://travis-ci.com/kernelmethod/LSH.jl.svg?branch=master)](https://travis-ci.com/kernelmethod/LSH.jl)
[![DOI](https://zenodo.org/badge/197700982.svg)](https://zenodo.org/badge/latestdoi/197700982)


Implementations of different [locality-sensitive hash functions](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) in Julia.

**Installation**: `julia> Pkg.add("https://github.com/kernelmethod/LSH.jl")`

So far, there are hash functions for the following measures of similarity:

- Cosine similarity
  - `SimHash`
- Jaccard similarity
  - `MinHash`
- `L^1` (Manhattan / "taxicab") and `L^2` (Euclidean) distance
  - `L1Hash`
  - `L2Hash`

This package still needs a lot of work, including improvement to the documentation and API. In general, if you want to draw one or more new hash functions, you can use the following syntax:

```julia
hashfn = HashFunctionFamily(number of hash functions;
                            [family-specific keyword params])
```
