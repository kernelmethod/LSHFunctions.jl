#================================================================

MonteCarloHash for hashing function spaces.

================================================================#

#========================
Typedefs
========================#

struct MonteCarloHash{H <: Union{SymmetricLSHFunction,AsymmetricLSHFunction}, D, T} <: LSHFunction
    discrete_hashfn :: H
    μ :: D
    sample_points :: Vector{T}
end

### External MonteCarloHash constructors

# TODO: restrict similarities. E.g. Jaccard should not be an available similarity
function MonteCarloHash(similarity, μ, args...; n_samples=1024, kws...)
    discrete_hashfn = LSHFunction(similarity, args...; kws...)
    sample_points = [μ() for ii = 1:n_samples]

    MonteCarloHash(discrete_hashfn, μ, sample_points)
end

#========================
MonteCarloHash helper functions
========================#

get_samples(hashfn::MonteCarloHash, f) = f.(hashfn.sample_points)

#========================
LSHFunction API compliance
========================#

hashtype(hashfn::MonteCarloHash) =
    hashtype(hashfn.discrete_hashfn)

similarity(hashfn::MonteCarloHash) =
    similarity(hashfn.discrete_hashfn)

n_hashes(hashfn::MonteCarloHash) =
    n_hashes(hashfn.discrete_hashfn)

# TODO: this may not be true
single_hash_collision_probability(hashfn::MonteCarloHash, args...; kws...) =
    single_hash_collision_probability(hashfn.discrete_hashfn, args...; kws...)

#========================
SymmetricLSHFunction API compliance
========================#

(hashfn::MonteCarloHash{<:SymmetricLSHFunction})(f) =
    index_hash(hashfn, f)

#========================
AsymmetricLSHFunction API compliance
========================#

function index_hash(hashfn::MonteCarloHash, f)
    samples = get_samples(hashfn, f)
    index_hash(hashfn.discrete_hashfn, samples)
end

function query_hash(hashfn::MonteCarloHash, f)
    samples = get_samples(hashfn, f)
    query_hash(hashfn.discrete_hashfn, samples)
end

#========================
MonteCarloHash API
========================#

# Compute the similarity between f and g in the embedded space
function embedded_similarity(hashfn::MonteCarloHash, f, g)
    samples_f = get_samples(hashfn, f)
    samples_g = get_samples(hashfn, g)
    simfun = similarity(hashfn)
    simfun(samples_f, samples_g)
end
