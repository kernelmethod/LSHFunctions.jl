#================================================================

Definition of MinHash, an LSH function for hashing on Jaccard similarity.

================================================================#

using Random: shuffle!

#========================
MinHash struct definition and constructors
========================#

struct MinHash{T, I <: Union{UInt32,UInt64}} <: SymmetricLSHFunction
    # Flag that indicates whether or not we were provided all of the symbols
    # we will possibly see in the MinHash constructor. This is used during
    # hashing to improve efficiency in the case where we were already provided
    # all of the symbols.
    fixed_symbols :: Bool

    # A list of dictionaries mapping symbols to integers. To compute a single
    # hash function, we pass every element of an input set through one of
    # the dictionaries and use the smallest output as our hash.
    mappings :: Vector{Dict{T,I}}
end

"""
    MinHash(n_hashes::Integer = 1;
            dtype::DataType = Any,
            symbols::Union{Vector,Set} = Set())

Construct a locality-sensitive hash function for Jaccard similarity.

# Arguments
- `n_hashes::Integer` (default: `1`): the number of hash functions to generate.

# Keyword parameters
- `dtype::DataType` (default: `Any`): the type of symbols in the sets you're hashing. This is overriden by the data type contained in `symbols` when `symbols` is non-empty.
- `symbols::Union{Vector,Set}`: a `Vector` or `Set` containing all of the possible elements ("symbols") of the sets that you will be hashing. If left empty, `MinHash` will instead expand its dictionary when it sees new symbols (at small additional computational expense).

# Examples
Construct a hash function to hash sets whose elements are integers between `1` and `50`:

```jldoctest; setup = :(using LSH)
julia> hashfn = MinHash(40; symbols = Set(1:50));

julia> n_hashes(hashfn) == 40 && similarity(hashfn) == jaccard
true

julia> hashfn(Set([38, 14, 29, 48, 11]));

julia> hashfn([1, 1, 2, 3, 4]); # You can also hash Vectors

julia> hashfn(Set([100]))
ERROR: Symbol 100 not found
```

If you aren't sure ahead of time exactly what kinds of elements will be in the sets you're hashing, you can opt not to specify `symbols`, in which case `MinHash` will lazily update its hash functions as it encounters new symbols:

```jldoctest; setup = :(using LSH)
julia> hashfn = MinHash();

julia> hashfn(Set([1, 2, 3]));

julia> hashfn(Set(["a", "b", "c"]));

```

If you don't know what elements you'll encounter, but you know that they'll all be of a specific data type, you can specify the `dtype` argument for increased efficiency:

```jldoctest; setup = :(using LSH)
julia> hashfn = MinHash(10; dtype = String);

julia> hashfn(Set(["a", "b", "c"]));

```

# References
```
Broder, A. "On the resemblance and containment of documents". Compression and Complexity of Sequences: Proceedings, Positano, Amalfitan Coast, Salerno, Italy, June 11-13, 1997. doi:10.1109/SEQUEN.1997.666900.
```

See also: [`jaccard`](@ref)
"""
function MinHash(args...;
                 dtype::DataType = Any,
                 symbols::C = Set{Any}()) where {T, C <: Union{Vector{T},Set{T}}}

    if length(symbols) > 0
        MinHash{T}(args...; symbols=symbols)
    else
        MinHash{dtype}(args...; symbols=Set{dtype}())
    end
end

function MinHash{T}(n_hashes::Integer = 1;
                    symbols::C = Set{T}()) where {T, C <: Union{Vector{<:T},Set{<:T}}}

    fixed_symbols = (length(symbols) > 0)

    # If fixed_symbols is true, and the symbol set is sufficiently small, then
    # we make our mappings from symbols -> integers map to UInt32 rather than
    # UInt64.
    # Note that we shouldn't do this if the symbol set is unspecified, because
    # in that case new mappings are determined randomly. With only 32  bits of
    # randomness, the probability that two symbols map to the same integer gets
    # high relatively quickly.
    hash_type =
        (fixed_symbols && length(symbols) ≤ typemax(UInt32)) ?
        UInt32 : UInt64

    mappings = Vector{Dict{T, hash_type}}(undef, n_hashes)

    if !fixed_symbols
        for ii = 1:length(mappings)
            mappings[ii] = Dict{T,hash_type}()
        end
    else
        # Create a new random mapping (i.e. a new Dict) that maps symbols to
        # integers in the range 1:length(symbols).
        mapping_range = convert.(hash_type, 1:length(symbols))

        for ii = 1:length(mappings)
            new_mapping = Dict{T,hash_type}()
            shuffle!(mapping_range)

            for (sym,h) in zip(symbols, mapping_range)
                new_mapping[sym] = h
            end

            mappings[ii] = new_mapping
        end
    end

    MinHash{T, hash_type}(fixed_symbols, mappings)
end

#========================
LSHFunction and SymmetricLSHFunction API compliance
========================#
n_hashes(hashfn :: MinHash) = length(hashfn.mappings)
hashtype(:: MinHash{T, I}) where {T, I} = I
similarity(::MinHash) = jaccard

single_hash_collision_probability(::MinHash, sim) = sim

LSH.@register_similarity!(jaccard, MinHash)

### Hash computation

function (hashfn :: MinHash{T,I})(x) where {T,I}
    # The iith hash is the smallest output of hashfn.mappings[ii] for all
    # inputs in the set x.
    if hashfn.fixed_symbols
        # If the symbols are fixed, throw an error whenever we encounter
        # a symbol we don't recognize.
        map(mapping ->
            minimum(get(mapping, xjj) do
                        "Symbol $(xjj) not found" |>
                        ErrorException |>
                        throw
                    end
                    for xjj in x),
            hashfn.mappings)
    else
        # If the symbols are non-fixed, assign new integer labels to
        # new symbols.
        map(mapping ->
            minimum(get!(mapping, xjj) do
                        rand(I)
                    end
                    for xjj in x),
            hashfn.mappings)
    end
end
