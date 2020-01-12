#================================================================

ChebHash for hashing the L^2([-1,1]) function space.

================================================================#

#========================
Typedefs
========================#

struct ChebHash{H <: LSHFunction}
    discrete_hashfn :: H
end

### External ChebHash constructors
ChebHash(similarity, args...; kws...) =
    ChebHash(SimilarityFunction(similarity), args...; kws...)

function ChebHash(::SimilarityFunction{S}, args...; kws...) where S
    discrete_hashfn = LSHFunction(S, args...; kws...)
    ChebHash(discrete_hashfn)
end

#========================
LSHFunction API compliance
========================#

n_hashes(hashfn::ChebHash) =
    n_hashes(hashfn.discrete_hashfn)

similarity(hashfn::ChebHash) =
    similarity(hashfn.discrete_hashfn)

hashtype(hashfn::ChebHash) =
    hashtype(hashfn.discrete_hashfn)

# TODO: this may not be true
single_hash_collision_probability(hashfn::ChebHash, args...; kws...) =
    single_hash_collision_probability(hashfn.discrete_hashfn, args...; kws...)
