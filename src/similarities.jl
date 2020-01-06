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

# Cosine similarity

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

# TODO: L^p distance

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
