#=============================================

Definitions of various similarity functions

=============================================#

using LinearAlgebra: dot, norm
using Markdown

#====================
A list of all of the available similarity functions that can be used for hashing
====================#

const available_similarities = Vector()

# Get a Vector of all of the available similarity functions as Strings
available_similarities_as_strings() = available_similarities .|> string |> sort

#====================
Definitions of built-in similarity functions
====================#

#=
Cosine similarity
=#

@doc raw"""
    CosSim(x,y)

Computes the cosine similarity between two inputs, `x` and `y`. Cosine similarity is defined as

```\math
CosSim(x,y) = \frac{\left\langle x,y\right\rangle}{\|x\|\cdot\|y\|}
```

where ``\left\langle\cdot,\cdot\right\rangle`` is an inner product (e.g. dot product) and ``\|\cdot\|`` is its derived norm. This is roughly interpreted as being related to the angle between the inputs `x` and `y`: when `x` and `y` have low angle between them, `CosSim(x,y)` is high (close to `1`). Meanwhile, when `x` and `y` have large angle between them, `CosSim(x,y)` is low (close to `-1`).

# Arguments
- `x` and `y`: two inputs for which `dot(x,y)`, `norm(x)`, and `norm(y)` are defined.

# Examples
```jldoctest; setup = :(using LSH)
julia> using LinearAlgebra: dot, norm;

julia> x, y = rand(4), rand(4);

julia> CosSim(x,y) == dot(x,y) / (norm(x) * norm(y))
true

julia> z = rand(5);

julia> CosSim(x,z)
ERROR: DimensionMismatch("dot product arguments have lengths 4 and 5")
```
"""
CosSim(x,y) = dot(x,y) / (norm(x) * norm(y))

push!(available_similarities, CosSim)

#=
L^p distance
=#

@doc raw"""
    ℓ_p(p, x, y)
    ℓ_1(x, y)
    ℓ_2(x, y)

Computes the ``\ell^p`` distance between a pair of vectors, given by

```\math
\ell^p(x,y) \coloneqq \|x - y\|_p = \sum \left|x_i - y_i\right|^p
```

Since ``\ell^1`` and ``\ell^2`` are both common cases of ``\ell^p`` distance, they are given unique function names `ℓ_1` and `ℓ_2` that you can use to call them.
"""
function ℓ_p(p::Integer, x::Vector{T}, y::Vector{T}) where {T}
    # TODO: more descriptive error message
    @assert p > 0
    @assert length(x) == length(y)

    result = T(0)
    @inbounds @simd for ii = 1:length(x)
        result += abs(x[ii] - y[ii])^p
    end

    return result^(1/p)
end

@doc (@doc ℓ_p)
function ℓ_1(x::Vector{T}, y::Vector{T}) where {T}
    # TODO: more descriptive error message
    @assert length(x) == length(y)

    result = T(0)
    @inbounds @simd for ii = 1:length(x)
        result += abs(x[ii] - y[ii])
    end

    return result
end

@doc (@doc ℓ_p)
function ℓ_2(x::Vector{T}, y::Vector{T}) where {T}
    # TODO: more descriptive error message
    @assert length(x) == length(y)
    result = T(0)

    @inbounds @simd for ii = 1:length(x)
        result += abs2(x[ii] - y[ii])
    end

    return √result
end

push!(available_similarities, ℓ_1)
push!(available_similarities, ℓ_2)

# TODO: Jaccard similarity

# TODO: inner product

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
