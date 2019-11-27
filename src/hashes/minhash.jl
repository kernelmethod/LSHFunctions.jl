using Random: shuffle!

#===================================

MinHash struct definition

====================================#


"""
MinHash implementation for Jaccard similarity-based hashing on sets. A `MinHash` struct takes as input a `Set` or `Vector` and computes one or more hashes for it. These hashes are locality-sensitive hashes for Jaccard similarity, defined as

    Sim(A, B) = length(A ∩ B) / length(A ∪ B)

References:

    [1] Broder, A. "On the resemblance and containment of documents". Compression and Complexity of Sequences: Proceedings, Positano, Amalfitan Coast, Salerno, Italy, June 11-13, 1997. doi:10.1109/SEQUEN.1997.666900. https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/broder97resemblance.pdf
    [2] https://en.wikipedia.org/wiki/MinHash
"""
struct MinHash{T, I <: Union{UInt32,UInt64}}
    fixed_symbols :: Bool
    mappings :: Vector{Dict{T,I}}
end

#===================================

MinHash constructors

====================================#

"""
    MinHash(
        symbols :: C,
        n_hashes :: Integer) where {T, C <: Union{Vector{T}, Set{T}}}

Sample hash function(s) from the MinHash family over the symbol set `symbols`.

## Arguments
- `symbols :: Union{Vector{T}, Set{T}}`: a `Vector` or `Set` containing all of the possible elements ("symbols") of the sets that you will be hashing.
- `n_hashes :: Integer`: the number of distinct hash functions that you want to create. The resulting `MinHash` struct will compute `n_hashes` hashes for each input that you provide.
"""
function MinHash(
        symbols :: C,
        n_hashes :: Integer) where {T, C <: Union{Vector{T}, Set{T}}}

    I = length(symbols) ≤ typemax(UInt32) ? UInt32 : UInt64
    possible_hashes = convert.(I, 1:length(symbols))
    mappings = Vector{Dict{T,I}}(undef, n_hashes)

    for ii = 1:length(mappings)
        shuffle!(possible_hashes)
        new_mapping = Dict{T,UInt32}()

        for (sym, h) in zip(symbols, possible_hashes)
            new_mapping[sym] = h
        end

        mappings[ii] = new_mapping
    end

    MinHash{T, I}(true, mappings)
end

"""
    MinHash(
        dtype :: DataType,
        n_hashes :: Integer)

Sample some new MinHash hash functions, where elements of the sets that you'll be hashing have type `dtype`. The `MinHash` struct returned by this constructor is lazily updated whenever it hashes a set with an element it's never seen before. Use this constructor for `MinHash` when you don't know ahead of time all of the possible elements of the sets that you'll be hashing.

Note that the `MinHash` struct returned by the `MinHash(::C, ::Integer) where {T, C <: Union{Vector{T}, Set{T}}}` constructor is generally more efficient. If you know in advance all of the possible elements of the sets that you'll be hashing, you should probably go with that constructor instead.

## Arguments
- `dtype :: DataType`: the type elements in the sets that you'll be hashing.
- `n_hashes :: Integer`: the number of distinct hash functions that you want to create. The resulting `MinHash` struct will compute `n_hashes` hashes for each input that you provide.
"""
function MinHash(
        dtype :: DataType,
        n_hashes :: Integer)

    mappings = Vector{Dict{dtype, UInt64}}(undef, n_hashes)
    for ii = 1:length(mappings)
        mappings[ii] = Dict{dtype, UInt64}()
    end

    MinHash{dtype, UInt64}(false, mappings)
end

"""
    MinHash(n_hashes :: Integer)

Alias for `MinHash(Any, :: Integer)`. Review the documentation for `MinHash(:: DataType, :: Integer)` for more information.
"""
MinHash(n_hashes :: Integer) = MinHash(Any, n_hashes)

#===================================

Implementation of hash computation

====================================#

function (hashfn :: MinHash)(x)
    if hashfn.fixed_symbols
        _minhash_hash_with_fixed_symbols(hashfn, x)
    else
        _minhash_hash_with_nonfixed_symbols(hashfn, x)
    end
end

function _minhash_hash_with_fixed_symbols(
        hashfn :: MinHash{T,I},
        x :: C) where {T, I, C <: Union{Set{T}, Vector{T}}}

    hashes = Vector{I}(undef, n_hashes(hashfn))

    for (ii, mapping) in enumerate(hashfn.mappings)
        # The value of the hash function corresponding to `mapping` is just the
        # element of `x` that has the smallest output under `mapping`.
        hashes[ii] = minimum(mapping[xjj] for xjj in x)
    end

    return hashes
end

function _minhash_hash_with_nonfixed_symbols(
        hashfn :: MinHash{T,I},
        x :: C) where {T, I, C <: Union{Set{<:T}, Vector{<:T}}}

    hashes = Vector{I}(undef, n_hashes(hashfn))

    for (ii, mapping) in enumerate(hashfn.mappings)
        # We compute the hash function the same way as we did in
        # _minhash_hash_with_fixed_symbols. However, we have to account for the
        # fact that the Set/Vector `x` may contain symbols that we've never seen
        # before. When that's the case, we have to assign a new integer to that
        # symbol, and update the `mapping` Dict.
        hashes[ii] = 
            minimum(
                get!(mapping, xjj) do 
                    rand(I)
                end 
                for xjj in x)
    end

    return hashes
end

#===================================

LSHFunction and SymmetricLSHFunction API compliance.

====================================#

n_hashes(hashfn :: MinHash) = length(hashfn.mappings)
hashtype(:: MinHash{T, I}) where {T, I} = I

function redraw!(hashfn :: MinHash)
    if hashfn.fixed_symbols
        _redraw_fixed_symbols!(hashfn)
    else
        _redraw_nonfixed_symbols!(hashfn)
    end
end

function _redraw_fixed_symbols!(hashfn :: MinHash)
    # Implementation of redraw! for MinHash when the symbol set is fixed.
    #
    # Since the symbol set is fixed, redrawing only amounts to rearranging the
    # mapping of keys to values. As a result, we can cheaply redraw the hash
    # function just by shuffling the mapping dictionaries' values.
    for mapping in hashfn.mappings
        shuffle!(mapping.vals)
    end
end

function _redraw_nonfixed_symbols!(hashfn :: MinHash{T,I}) where {T, I}
    # Implementation of redraw! for MinHash when the symbol set is non-fixed.
    #
    # We assume that we won't see the same set of symbols again (or that we will
    # see very different symbols) after redrawing. Under that assumption, it makes
    # more sense to simply clear out all of the mappings, and lazily update them
    # when we see new symbols.
    for ii = 1:length(hashfn.mappings)
        hashfn.mappings[ii] = Dict{T,I}()
    end
end
