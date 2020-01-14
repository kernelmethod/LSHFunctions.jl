#=============================================

Definitions of various similarity functions

=============================================#

using Markdown, QuadGK
using LinearAlgebra: dot, norm

#====================
Definitions of built-in similarity functions
====================#

#====================
Cosine similarity
====================#

@doc raw"""
    cossim(x,y)

Computes the cosine similarity between two inputs, `x` and `y`. Cosine similarity is defined as

```\math
cossim(x,y) = \frac{\left\langle x,y\right\rangle}{\|x\|\cdot\|y\|}
```

where ``\left\langle\cdot,\cdot\right\rangle`` is an inner product (e.g. dot product) and ``\|\cdot\|`` is its derived norm. This is roughly interpreted as being related to the angle between the inputs `x` and `y`: when `x` and `y` have low angle between them, `cossim(x,y)` is high (close to `1`). Meanwhile, when `x` and `y` have large angle between them, `cossim(x,y)` is low (close to `-1`).

# Arguments
- `x` and `y`: two inputs for which `dot(x,y)`, `norm(x)`, and `norm(y)` are defined.

# Examples
```jldoctest; setup = :(using LSH)
julia> using LinearAlgebra: dot, norm;

julia> x, y = rand(4), rand(4);

julia> cossim(x,y) == dot(x,y) / (norm(x) * norm(y))
true

julia> z = rand(5);

julia> cossim(x,z)
ERROR: DimensionMismatch("dot product arguments have lengths 4 and 5")
```

See also: [`SimHash`](@ref)
"""
cossim(x::AbstractVector, y::AbstractVector) = dot(x,y) / (norm(x) * norm(y))

function cossim(f, g, interval::LSH.RealInterval)
    norm_f = L2_norm(f, interval)
    norm_g = L2_norm(g, interval)
    inner_prod(f, g, interval) / (norm_f * norm_g)
end

#====================
L^p distance
====================#

@doc raw"""
    ℓp(p, x, y)
    ℓ1(x, y)
    ℓ2(x, y)

Computes the ``\ell^p`` distance between a pair of vectors, given by

```\math
\ell^p(x,y) \coloneqq \|x - y\|_p = \sum \left|x_i - y_i\right|^p
```

Since ``\ell^1`` and ``\ell^2`` are both common cases of ``\ell^p`` distance, they are given unique function names `ℓ1` and `ℓ2` that you can use to call them.
"""
ℓp(x::AbstractVector, y::AbstractVector, p::Real=2) = Lp(x, y, p)

@doc (@doc ℓp)
ℓ1(x::AbstractVector, y::AbstractVector)            = L1(x, y)

@doc (@doc ℓp)
ℓ2(x::AbstractVector, y::AbstractVector)            = L2(x, y)

function Lp(x::AbstractVector{T}, y::AbstractVector{T}, p::Real=2) where {T}
    # TODO: more descriptive error message
    @assert p > 0
    @assert size(x) == size(y)

    result = T(0)
    @inbounds @simd for ii = 1:length(x)
        result += abs(x[ii] - y[ii])^p
    end

    return result^(1/p)
end

function L1(x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    # TODO: more descriptive error message
    @assert length(x) == length(y)

    result = T(0)
    @inbounds @simd for ii = 1:length(x)
        result += abs(x[ii] - y[ii])
    end

    return result
end

function L2(x::AbstractVector{T}, y::AbstractVector{T}) where {T}
    # TODO: more descriptive error message
    @assert length(x) == length(y)
    result = T(0)

    @inbounds @simd for ii = 1:length(x)
        result += abs2(x[ii] - y[ii])
    end

    return √result
end

# Function space L^p distances
Lp(f, g, p::Real=2) = Lp_norm(x -> f(x) - g(x), p)
L1(f, g)            = L1_norm(x -> f(x) - g(x))
L2(f, g)            = L2_norm(x -> f(x) - g(x))

#====================
Jaccard similarity
====================#

@doc raw"""
    jaccard(A::Set, B::Set) :: Float64

Computes the Jaccard similarity between sets `A` and `B`, which is defined as

```\math
J(A,B) = \frac{\left|A \cap B\right|}{\left|A \cup B\right|}
```

# Arguments
- `A::Set`, `B::Set`: the two sets with which to compute Jaccard similarity.

# Returns
`Float64`: the Jaccard similarity between sets `A` and `B`, which is between `0` and `1`.

# Examples
```jldoctest; setup = :(using LSH)
julia> A, B = Set([1, 2, 3]), Set([2, 3, 4]);

julia> jaccard(A,B)
0.5

julia> jaccard(A,B) == length(A ∩ B) / length(A ∪ B)
true
```

See also: [`MinHash`](@ref)
"""
function jaccard(A::Set, B::Set) :: Float64
    # To avoid corner cases where A and B are both empty
    if isempty(A)
        Float64(0)
    else
        length(A ∩ B) / length(A ∪ B)
    end
end

#====================
Inner product and norms
====================#

### Inner products
# TODO: docs
inner_prod(x::AbstractVector, y::AbstractVector) = dot(x,y)

# 1-dimensional inner product between L^2 functions
inner_prod(f, g, interval::LSH.RealInterval) =
    quadgk(x -> f(x)g(x), interval.lower, interval.upper)[1]

### L^p norms
Lp_norm(x::AbstractVector, p::Real = 2) = norm(x,p)
L1_norm(x::AbstractVector)              = norm(x,1)
L2_norm(x::AbstractVector)              = norm(x)

ℓp_norm(x::AbstractVector, p::Real = 2) = Lp_norm(x, p)
ℓ1_norm(x::AbstractVector)              = L1_norm(x)
ℓ2_norm(x::AbstractVector)              = L2_norm(x)

# 1-dimensional L^p norms
Lp_norm(f, interval::LSH.RealInterval, p::Real=2) = quadgk(x -> abs(f(x)).^p, interval.lower, interval.upper)[1]
L1_norm(f, interval::LSH.RealInterval)            = quadgk(x -> abs(f(x)),    interval.lower, interval.upper)[1]
L2_norm(f, interval::LSH.RealInterval)            = quadgk(x -> abs2(f(x)),   interval.lower, interval.upper)[1]

#====================
1D Wasserstein distance
====================#

@doc raw"""
    wasserstein_1d(f, g, p)
    wasserstein1_1d(f, g)
    wasserstein2_1d(f, g)
    emd(f, g)

Compute the order-``p`` Wasserstein distance between two probability distributions defined on the interval ``[-1,1]``.
- `wasserstein1_1d(f,g) == emd(f,g) == wasserstein_1d(f, g, 1)`
- `wasserstein2_1d(f,g) == wasserstein_1d(f, g, 2)`

# Arguments
- `f` and `g`: two probability distributions defined on ``[-1,1]``.
- `p::AbstractFloat`: the order of Wasserstein distance to compute.

# Returns
The order-``p`` Wasserstein distance between `f` and `g` (an `AbstractFloat`).
"""
function wasserstein_1d(f, g, p::AbstractFloat)
    # For one-dimensional probability distributions, the Wasserstein distance has the
    # closed form
    #
    #       ∫_0^1 |F^{-1}(x) - G^{-1}(x)|^p dx
    #
    # where F^{-1} and G^{-1} are the inverse c.d.f.s of f and g. We use this
    # formula to compute the distance between f and g.
    error("TODO")
end

wasserstein1_1d(f, g) = wasserstein_1d(f, g, 1)
wasserstein2_2d(f, g) = wasserstein_1d(f, g, 2)
emd(f, g)             = wasserstein1_1d(f, g)

#====================
Definitions for similarity function-related components of the AbstractLSHFunction
API.
====================#

# Define documentation for `similarity` manually so that we can dynamically
# modify it through the available_similarities list.
Docs.getdoc(::typeof(similarity)) = Markdown.parse("""
    similarity(hashfn::AbstractLSHFunction)

Returns the similarity function that the input `AbstractLSHFunction` hashes on.

# Arguments
- `hashfn::AbstractLSHFunction`: the hash function whose similarity we would like to retrieve.

# Returns
    Returns a similarity function, which is one of the following:

```
$(join(available_similarities_as_strings(), "\n"))
```
""")
