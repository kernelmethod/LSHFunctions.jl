#================================================================

MonteCarloHash for hashing function spaces.

================================================================#

#========================
Typedefs
========================#

struct MonteCarloHash{H <: Union{SymmetricLSHFunction,AsymmetricLSHFunction},
                      D, T, S} <: LSHFunction
    discrete_hashfn :: H
    μ :: D

    # TODO: make typeof(volume) and typeof(p) match the element type
    # returned by the sampler μ.

    # The volume of the L^p space we're embedding, defined as
    #
    #   volume = ∫_Ω dx
    #
    volume :: T

    # The order of the L^p space that we're embedding
    p :: T

    n_samples :: Int64
    sample_points :: Vector{S}

    ### Internal constructors
    function MonteCarloHash(discrete_hashfn::H, μ, volume, p, n_samples) where {H<:LSHFunction}
        sample_points = [μ() for ii = 1:n_samples]

        T = eltype(μ())
        volume = T(volume)
        p = T(p)

        new{H,typeof(μ),T,eltype(sample_points)}(discrete_hashfn, μ, volume, p, n_samples, sample_points)
    end
end

### External MonteCarloHash constructors

# TODO: restrict similarities. E.g. Jaccard should not be an available similarity
@generated function MonteCarloHash(similarity, μ, args...;
                                   n_samples::Int64 = 1024, volume=1.0, kws...)

    p = begin
        if similarity <: Union{typeof(cossim),typeof(ℓ_2)}
            :(p = 2.0)
        elseif similarity <: typeof(ℓ_1)
            :(p = 1.0)
        else
            quote
                "similarity must be cossim, ℓ_1, or ℓ_2" |>
                ErrorException |>
                throw
            end
        end
    end

    quote
        discrete_hashfn = LSHFunction(similarity, args...; kws...)
        p = $(esc(p))
        MonteCarloHash(discrete_hashfn, μ, volume, p, n_samples)
    end
end

#========================
MonteCarloHash helper functions
========================#

# Embed an input function f in the discrete space
function get_samples(hashfn::MonteCarloHash, f)
    α = (hashfn.volume / hashfn.n_samples)^(1/hashfn.p)
    α * f.(hashfn.sample_points)
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
