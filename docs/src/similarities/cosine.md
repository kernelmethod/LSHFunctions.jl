# Cosine similarity

!!! warning "Under construction"
    This section is currently being developed. If you're interested in helping write this section, feel free to [open a pull request](https://github.com/kernelmethod/LSH.jl/pulls); otherwise, please check back later.

## Definition
[Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity), roughly speaking, measures the angle between a pair of inputs. Two inputs are very similar if the angle between them is low, and their similarity drops as the angle between them increases.

Concretely, cosine similarity is computed as

``\text{cossim}(x,y) = \frac{\left\langle x,y\right\rangle}{\|x\|\cdot\|y\|} = \left\langle\frac{x}{\|x\|},\frac{y}{\|y\|}\right\rangle``

where ``\left\langle\cdot,\cdot\right\rangle`` is an inner product (e.g., dot product) and ``\|\cdot\|`` is the norm derived from that inner product. ``\text{cossim}(x,y)`` goes from ``-1`` to ``1``, where ``-1`` corresponds to low similarity and ``1`` corresponds to high similarity. To calculate cosine similarity, you can use the [`cossim`](@ref) function exported from the `LSH` module:

```jldoctest
julia> using LSH, LinearAlgebra

julia> x = [5, 3, -1, 1];  # norm(x) == 6

julia> y = [2, -2, -2, 2]; # norm(y) == 4

julia> cossim(x,y) == dot(x,y) / (norm(x)*norm(y))
true

julia> cossim(x,y) == (5*2 + 3*(-2) + (-1)*(-2) + 1*2) / (6*4)
true
```

## SimHash
*SimHash*[^1][^2] is a family of LSH functions for hashing with respect to cosine similarity. You can generate a new hash function from this family by calling [`SimHash`](@ref):

```jldoctest; setup = :(using LSH)
julia> hashfn = SimHash();

julia> n_hashes(hashfn)
1

julia> hashfn = SimHash(40);

julia> n_hashes(hashfn)
40
```

Once constructed, you can start hashing vectors by calling `hashfn(x)`:

```jldoctest; setup = :(using LSH, Random; Random.seed!(0)), output = false
hashfn = SimHash(100)

# x and y have high cosine similarity since they point in the same direction
# x and z have low cosine similarity since they point in opposite directions
x = randn(128)
y = 2x
z = -x

hx, hy, hz = hashfn(x), hashfn(y), hashfn(z)

# Among the 100 hash functions that we generated, we expect more hash
# collisions between x and y than between x and z
sum(hx .== hy) > sum(hx .== hz)

# output
true

```

Note that [`SimHash`](@ref) is a one-bit hash function. As a result, `hashfn(x)` returns a `BitArray`:

```jldoctest; setup = :(using LSH)
julia> hashfn = SimHash();

julia> n_hashes(hashfn)
1

julia> hashes = hashfn(randn(4));

julia> typeof(hashes)
BitArray{1}

julia> length(hashes)
1
```

Since a single-bit hash doesn't do much to reduce the cost of similarity search, you usually want to generate multiple hash functions at once. For instance, in the snippet below we sample 10 hash functions, so that `hashfn(x)` is a length-10 `BitArray`:

```jldoctest; setup = :(using LSH)
julia> hashfn = SimHash(10);

julia> n_hashes(hashfn)
10

julia> hashes = hashfn(randn(4));

julia> length(hashes)
10
```

The probability of a hash collision (for a single hash) is

``Pr[h(x) = h(y)] = 1 - \frac{\theta}{\pi}``

where ``\theta = \text{arccos}(\text{cossim}(x,y))`` is the angle between ``x`` and ``y``. This collision probability is shown in the plot below.

```@eval
using PyPlot, LSH
hashfn = SimHash()
x = range(-1, 1; length=1024)
y = [LSH.single_hash_collision_probability(hashfn, xii) for xii in x]

plot(x, y)
title("Probability of hash collision for SimHash")
xlabel(raw"$cossim(x,y)$")
ylabel(raw"$Pr[h(x) = h(y)]$")

savefig("simhash_collision_probability.svg")
```

![Probability of collision for SimHash](simhash_collision_probability.svg)

### Footnotes

[^1]: Moses S. Charikar. *Similarity estimation techniques from rounding algorithms*. In Proceedings of the Thiry-Fourth Annual ACM Symposium on Theory of Computing, STOC '02, page 380–388, New York, NY, USA, 2002. Association for Computing Machinery. 10.1145/509907.509965.

[^2]: [`SimHash` API reference](@ref SimHash)
