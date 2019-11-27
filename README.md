# LSH.jl
Implementations of different [locality-sensitive hash functions](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) in Julia.

**Installation**: `julia> Pkg.add("https://github.com/kernelmethod/LSH.jl")`

So far, there are hash functions for the following measures of similarity:

- Cosine similarity
  - `SimHash`
- Jaccard similarity
  - `MinHash`
- `L^1` (Manhattan / "taxicab") and `L^2` (Euclidean) distance
  - `LpDistanceHash`
  - `L1Hash`
  - `L2Hash`
- Inner product magnitude (for maximum inner product search)
  - `MIPSHash`
  - `SignALSH` (recommended)

In addition, this module provides `LSHTable` and `LSHTableGroup` composite types to make it easier to build hash tables and groups of hash tables for use with LSH.
