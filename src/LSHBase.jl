#================================================================

Common typedefs and functions used throughout the LSH module.

================================================================#

#========================
Global variables and constants
========================#

# available_similarities: a set of all of the similarity functions that have
# been associated with hash functions via the register_similarity! macro.
const available_similarities = Set()

# Defaults to use for common arguments
const DEFAULT_N_HASHES = 1
const DEFAULT_DTYPE = Float32
const DEFAULT_RESIZE_POW2 = false

#========================
Abstract typedefs
========================#

abstract type LSHFunction end

@doc """
    abstract type SymmetricLSHFunction <: LSHFunction end

A symmetric locality-sensitive hashing function. A `SymmetricLSHFunction` uses the same hash function to insert items into a hash table as well as query the table for collisions. If `hashfn` is a `SymmetricLSHFunction`, you can compute the hash for an input `x` as `hashfn(x)`.

See also: [`AsymmetricLSHFunction`](@ref)
"""
abstract type SymmetricLSHFunction <: LSHFunction end

@doc """
    abstract type AsymmetricLSHFunction <: LSHFunction end

An asymmetric locality-sensitive hashing function. An `AsymmetricLSHFunction` uses one hash function to insert items into a hash table, and a different hash function to query the table for collisions. If `hashfn` is an `AsymmetricLSHFunction`, you can compute the indexing hash for an input `x` with `index_hash(hashfn,x)`, and the querying hash with `query_hash(hashfn,x)`.

See also: [`SymmetricLSHFunction`](@ref)
"""
abstract type AsymmetricLSHFunction <: LSHFunction end

#========================
Similarity function API
========================#

# Value type to encode different similarity functions
struct SimilarityFunction{F} end
SimilarityFunction(sim) = SimilarityFunction{sim}()

#========================
LSHFunction API
========================#

macro register_similarity! end
function LSHFunction end
function lsh_family end

@doc """
    collision_probability(hashfn::H, sim;
                          n_hashes::Union{Symbol,Integer}=:auto) where {H <: LSHFunction}

Compute the probability of hash collision between two inputs with similarity `sim` for an [`LSHFunction`](@ref) of type `H`. This function returns the probability that `n_hashes` hashes simultaneously collide.

# Arguments
- `hashfn::LSHFunction`: the `LSHFunction` for which we want to compute the probability of collision.
- `sim`: a similarity (or vector of similarities), computed using the similarity function returned by `similarity(hashfn)`.

# Keyword arguments
- `n_hashes::Union{Symbol,Integer}` (default: `:auto`): the number of hash functions to use to compute the probability of collision. If the probability that a single hash collides is `p`, then the probability that `n_hashes` hashes simultaneously collide is `p^n_hashes`. As a result,

  ```
  collision_probability(hashfn, sim; n_hashes=N)
  ```

  is the same as

  ```
  collision_probability(hashfn, sim; n_hashes=1).^N
  ```

  If `n_hashes = :auto` then this function will select the number of hashes to be `n_hashes(hashfn)` (using the [`n_hashes`](@ref) function from the [`LSHFunction`](@ref) API).

# Examples
The probability that a single MinHash hash function causes a hash collision between inputs `A` and `B` is equal to `jaccard(A,B)`:

```jldoctest; setup = :(using LSHFunctions)
julia> hashfn = MinHash();

julia> A = Set(["a", "b", "c"]);

julia> B = Set(["b", "c", "d"]);

julia> jaccard(A,B)
0.5

julia> collision_probability(hashfn, jaccard(A,B); n_hashes=1)
0.5
```

If our [`MinHash`](@ref) struct keeps track of `N` hash functions simultaneously, then the probability of collision is `jaccard(A,B)^N`:

```jldoctest; setup = :(using LSHFunctions)
julia> hashfn = MinHash(10);

julia> A = Set(["a", "b", "c"]);

julia> B = Set(["b", "c", "d"]);

julia> collision_probability(hashfn, jaccard(A,B)) ==
       collision_probability(hashfn, jaccard(A,B); n_hashes=10) ==
       collision_probability(hashfn, jaccard(A,B); n_hashes=1)^10
true
```

See also: [`n_hashes`](@ref), [`similarity`](@ref)
"""
@generated function collision_probability(hashfn::LSHFunction, sim;
                                          n_hashes::Union{Symbol,Integer} = :auto)

    error_msg = :("n_hashes must be :auto or a positive Integer" |>
                  ErrorException |>
                  throw)

    n_hashes = begin
        if n_hashes <: Symbol
            quote
                if n_hashes != :auto
                    $error_msg
                end

                n_hashes = _n_hashes(hashfn)
            end
        else
            quote
                if n_hashes ≤ 0
                    $error_msg
                end
                nh = n_hashes
            end
        end
    end

    quote
        $n_hashes
        single_hash_collision_probability(hashfn, sim).^n_hashes
    end
end

@doc """
    collision_probability(hashfn::LSHFunction, x, y;
                          n_hashes::Union{Symbol,Integer} = :auto)

Computes the probability of a hash collision between two inputs `x` and `y` for a given hash function `hashfn`. This is the same as calling

    collision_probability(hashfn, similarity(hashfn)(x,y); n_hashes=n_hashes)

# Examples
The following snippet computes the probability of collision between two sets `A` and `B` for a single MinHash. For MinHash, this probability is just equal to the Jaccard similarity between `A` and `B`.

```jldoctest; setup = :(using LSHFunctions)
julia> hashfn = MinHash();

julia> A = Set(["a", "b", "c"]);

julia> B = Set(["a", "b", "c"]);

julia> similarity(hashfn) == jaccard
true

julia> collision_probability(hashfn, A, B) ==
       collision_probability(hashfn, jaccard(A,B)) ==
       jaccard(A,B)
true
```

We can use the `n_hashes` argument to specify the probability that `n_hashes` MinHash hash functions simultaneously collide. If left unspecified, then we'll simply use `n_hashes(hashfn)` as the number of hash functions:

```jldoctest; setup = :(using LSHFunctions)
julia> hashfn = MinHash(10);

julia> A = Set(["a", "b", "c"]);

julia> B = Set(["a", "b", "c"]);

julia> collision_probability(hashfn, A, B) ==
       collision_probability(hashfn, A, B; n_hashes=10) ==
       collision_probability(hashfn, A, B; n_hashes=1)^10
true
```
"""
collision_probability(hashfn::LSHFunction, A, B; kws...) =
    collision_probability(hashfn, similarity(hashfn)(A,B); kws...)

#=
The following functions must be defined for all LSHFunction subtypes
=#

@doc """
    similarity(hashfn::LSHFunction)

Returns the similarity function that `hashfn` hashes on.

# Arguments
- `hashfn::AbstractLSHFunction`: the hash function whose similarity we would like to retrieve.

# Examples
```jldoctest; setup = :(using LSHFunctions)
julia> hashfn = LSHFunction(cossim);

julia> similarity(hashfn) == cossim
true

julia> hashfn = LSHFunction(ℓ1);

julia> similarity(hashfn) == ℓ1
true
```
"""
function similarity end

@doc """
    hashtype(hashfn::LSHFunction)

Returns the type of hash generated by a hash function.

# Examples
```jldoctest; setup = :(using LSHFunctions)
julia> hashfn = LSHFunction(cossim);

julia> hashtype(hashfn)
Bool

julia> hashfn = LSHFunction(ℓ1);

julia> hashtype(hashfn)
Int32
```
"""
function hashtype end

@doc """
    n_hashes(hashfn::LSHFunction)

Return the number of hashes computed by `hashfn`.

# Examples
```jldoctest; setup = :(using LSHFunctions)
julia> hashfn = SimHash();

julia> n_hashes(hashfn)
$(DEFAULT_N_HASHES)

julia> hashfn = SimHash(12);

julia> n_hashes(hashfn)
12

julia> hashes = hashfn(rand(25));

julia> length(hashes)
12
```
"""
function n_hashes end

# Alias for n_hashes that's occasionally useful when we need to process
# variables that are named n_hashes
const _n_hashes = n_hashes

# The function
#
#       single_hash_collision_probability(hashfn::H, sim)
#
# must be implemented for every subtype H of LSHFunction. Note that users don't
# access this function directly; instead, they use the collision_probability
# function exported by the LSH API.
function single_hash_collision_probability end

#========================
SymmetricLSHFunction API
========================#

@doc """
    index_hash(hashfn::SymmetricLSHFunction, x)

Identical to calling `hashfn(x)`.

See also: [`query_hash`](@ref), [`SymmetricLSHFunction`](@ref)
"""
index_hash(hashfn::SymmetricLSHFunction, x) = hashfn(x)

@doc """
    query_hash(hashfn::SymmetricLSHFunction, x)

Identical to calling `hashfn(x)`.

See also: [`index_hash`](@ref), [`SymmetricLSHFunction`](@ref)
"""
query_hash(hashfn::SymmetricLSHFunction, x) = hashfn(x)

#=
The following functions must be defined for all SymmetricLSHFunction subtypes
=#
function (::SymmetricLSHFunction) end

#========================
Abstract typedefs
========================#

#=
The following functions must be defined for all AsymmetricLSHFunction subtypes
=#

@doc """
    index_hash(hashfn::AsymmetricLSHFunction, x)

Computes the indexing hash (the hash used to insert items into the hash table) for an `AsymmetricLSHFunction` with input `x`.

See also: [`query_hash`](@ref), [`AsymmetricLSHFunction`](@ref)
"""
function index_hash end

@doc """
    query_hash(hashfn::AsymmetricLSHFunction, x)

Computes the querying hash (the hash used to query for items in the hash table) for an `AsymmetricLSHFunction` with input `x`.

See also: [`index_hash`](@ref), [`AsymmetricLSHFunction`](@ref)
"""
function query_hash end

#========================
Documentation utilities
========================#

available_similarities_as_strings() = available_similarities .|> string |> sort

### Docstring generators for common keyword arguments
N_HASHES_DOCSTR(; default = DEFAULT_N_HASHES) = """
`n_hashes::Integer` (default: `$(default)`): the number of hash functions to generate."""

DTYPE_DOCSTR(hashfn; default = DEFAULT_DTYPE) = """
`dtype::DataType` (default: `$(default)`): the data type to use in the $(hashfn) internals. For performance reasons you should pick `dtype` to match the type of the data you're hashing."""

RESIZE_POW2_DOCSTR(hashfn; default = DEFAULT_RESIZE_POW2) = """
`resize_pow2::Bool` (default: `$(default)`): affects the way in which the returned `$(hashfn)` resizes to hash inputs of different sizes. If you think you'll be hashing inputs of many different sizes, it's more efficient to set `resize_pow2 = true`."""

