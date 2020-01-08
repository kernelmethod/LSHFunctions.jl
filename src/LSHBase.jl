#================================================================

Common typedefs and functions used throughout the LSH module.

================================================================#

#========================
Global variables and constants
========================#

# available_similarities: a set of all of the similarity functions that have
# been associated with hash functions via the register_similarity! macro.
const available_similarities = Set()

available_similarities_as_strings() = available_similarities .|> string |> sort

#========================
Abstract typedefs
========================#
abstract type LSHFunction end
abstract type SymmetricLSHFunction <: LSHFunction end
abstract type AsymmetricLSHFunction <: LSHFunction end

#========================
LSHFunction API
========================#

@doc """
    register_similarity!(similarity, hashfn)

Register `hashfn` to the `LSH` module as the default locality-sensitive hash function to use for similarity function `similarity`. This makes it possible to construct a new hash function for `similarity` with `LSHFunction(similarity, args...; kws...)`.

# Arguments
- `similarity`: the similarity function to register.
- `hashfn`: the default locality-sensitive hash function that `similarity` should be associated with.

# Examples
Create a custom implementation of cosine similarity called `my_cossim`, and associate it with `SimHash`:

```jldoctest; setup = :(using LSH)
julia> using LinearAlgebra: dot, norm

julia> my_cossim(x,y) = dot(x,y) / (norm(x) * norm(y));

julia> hashfn = LSHFunction(my_cossim);
ERROR: MethodError: no method matching LSHFunction(::typeof(my_cossim))

julia> LSH.@register_similarity!(my_cossim, SimHash);

julia> hashfn = LSHFunction(my_cossim);

julia> isa(hashfn, SimHash)
true
```
"""
macro register_similarity!(similarity, hashfn)
    fn = :(LSH.LSHFunction)

    quote
        local similarity = $(esc(similarity))

        # Check that similarity is actually callable
        if similarity |> methods |> length == 0
            "similarity must be callable" |>
            ErrorException |>
            throw
        end

        if similarity ∈ available_similarities
            "Similarity function " * string(similarity) * " has already been registered." |>
            ErrorException |>
            throw
        end

        union!(available_similarities, Set([similarity]))

        # Define LSHFunction(similarity, args...; kws...)
        $(esc(fn))(::typeof($(esc(similarity))), args...; kws...) =
            $(esc(hashfn))(args...; kws...)
    end
end

#=
The following functions must be defined for all LSHFunction subtypes
=#
function hashtype end
function n_hashes end
function similarity end

#========================
SymmetricLSHFunction API
========================#

index_hash(h :: SymmetricLSHFunction, x) = h(x)
query_hash(h :: SymmetricLSHFunction, x) = h(x)

#=
The following functions must be defined for all SymmetricLSHFunction subtypes
=#
function (:: SymmetricLSHFunction) end

#========================
Abstract typedefs
========================#

#=
The following functions must be defined for all AsymmetricLSHFunction subtypes
=#
function index_hash end
function query_hash end
