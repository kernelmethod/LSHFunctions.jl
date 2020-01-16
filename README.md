# LSH.jl

- Build status: [![Build Status](https://travis-ci.com/kernelmethod/LSH.jl.svg?branch=master)](https://travis-ci.com/kernelmethod/LSH.jl)
- Code coverage: [![Coverage Status](https://coveralls.io/repos/github/kernelmethod/LSH.jl/badge.svg?branch=master)](https://coveralls.io/github/kernelmethod/LSH.jl?branch=master)
[![codecov](https://codecov.io/gh/kernelmethod/LSH.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kernelmethod/LSH.jl)
- DOI to cite this code: [![DOI](https://zenodo.org/badge/197700982.svg)](https://zenodo.org/badge/latestdoi/197700982)

Implementations of different [locality-sensitive hash functions](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) in Julia.

**Installation**: `julia> Pkg.add("https://github.com/kernelmethod/LSH.jl")`

So far, there are hash functions for the following measures of similarity:

- Cosine similarity (`SimHash`)
- Jaccard similarity (`MinHash`)
- L1 (Manhattan / "taxicab") distance: `L1Hash`
- L2 (Euclidean) distance: `L2Hash`
- Inner product
  - `SignALSH` (recommended)
  - `MIPSHash`
- Function-space hashes
  - `MonteCarloHash` (supports L1, L2, and cosine similarity)
  - `ChebHash` (supports L1, L2, and cosine similarity)

This package still needs a lot of work, including improvement to the documentation and API. In general, if you want to draw one or more new hash functions, you can use the following syntax:

```julia
hashfn = LSHFunction(similarity; [LSH family-specific keyword arguments])
```
