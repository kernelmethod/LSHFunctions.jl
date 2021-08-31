# Jaccard similarity

!!! warning "Under construction"
    This section is currently being developed. If you're interested in helping write this section, feel free to [open a pull request](https://github.com/kernelmethod/LSHFunctions.jl/pulls); otherwise, please check back later.

## Definition
*Jaccard similarity* is a statistic that measures the amount of overlap between two sets. It is defined as

```math
J(A,B) = \frac{|A \cap B|}{|A \cup B|}
```

``J(A,B)`` is bounded by ``0 \le J(A,B) \le 1``, with values close to 1 indicating high similarity and values close to 0 indicating low similarity.

You can calculate Jaccard similarity with the LSHFunctions package by calling [`jaccard`](@ref):

```jldoctest
julia> using LSHFunctions;

julia> A = Set([1, 2, 3]); B = Set([2, 3, 4]);

julia> jaccard(A,B) ==
       length(A ∩ B) / length(A ∪ B) ==
       0.5
true
```

## MinHash
*MinHash*[^Broder97] is a hash function for Jaccard similarity. It takes as input a set, and returns as output a `UInt32` or a `UInt64`. To sample a function from the MinHash LSH family, simply call [`MinHash`](@ref) with the number of hash functions you want to generate:

```jldoctest; setup = :(using LSHFunctions, Random; Random.seed!(0))
julia> hashfn = MinHash(5);

julia> n_hashes(hashfn)
5

julia> hashtype(hashfn)
UInt64

julia> A = Set([1, 2, 3]);

julia> hashfn(A)
5-element Vector{UInt64}:
 0x68ab426365cf3fcf
 0x13095267e4625e58
 0x0c0f74c97d4d341e
 0x1b294d39ad1e06a7
 0x74a5bce47c6b635a
```

The probability of a collision for an individual hash between sets ``A`` and ``B`` is just equal to their Jaccard similarity, i.e.

```math
Pr[h(A) = h(B)] = J(A,B)
```

```@eval
using PyPlot, LSHFunctions;
fig = figure();
hashfn = MinHash();
x = range(0, 1; length=1024);
y = collision_probability(hashfn, x; n_hashes=1);

plot(x, y)
title("Probability of hash collision for MinHash")
xlabel(raw"$J(A,B)$")
ylabel(raw"$Pr[h(x) = h(y)]$")

savefig("minhash_collision_probability.svg")
```

![Probability of collision for MinHash](minhash_collision_probability.svg)

## Footnotes
[^Broder97]: Broder, A. *On the resemblance and containment of documents*. Compression and Complexity of Sequences: Proceedings, Positano, Amalfitan Coast, Salerno, Italy, June 11-13, 1997. doi:10.1109/SEQUEN.1997.666900.
