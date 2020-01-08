#================================================================

MonteCarloHash for hashing function spaces.

================================================================#

#========================
Typedefs
========================#

struct MonteCarloHash{H <: Union{SymmetricLSHFunction,AsymmetricLSHFunction}, D} <: LSHFunction
    discrete_hashfn :: H
    μ :: D
end

### External MonteCarloHash constructors

function MonteCarloHash(similarity, μ, args...; kws...)
    discrete_hashfn = LSHFunction(similarity, args...; kws...)
    MonteCarloHash(discrete_hashfn, μ)
end

#========================
LSHFunction API compliance
========================#

hashtype(hashfn::MonteCarloHash) =
    hashtype(hashfn.discrete_hashfn)

similarity(hashfn::MonteCarloHash) =
    similarity(hashfn.discrete_hashfn)

n_hashes(hashfn::MonteCarloHash) =
    n_hashes(hashfn.discrete_hashfn)

#========================
SymmetricLSHFunction API compliance
========================#

function (hashfn::MonteCarloHash{<:SymmetricLSHFunction})(x)
end

#========================
AsymmetricLSHFunction API compliance
========================#

function index_hash(hashfn::MonteCarloHash, x)
end

function query_hash(hashfn::MonteCarloHash, x)
end




