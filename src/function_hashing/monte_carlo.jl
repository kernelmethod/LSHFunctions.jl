#================================================================

MonteCarloHash for hashing function spaces.

================================================================#

#========================
Typedefs
========================#

struct MonteCarloHash{F, H <: LSHFunction, D, T, S} <: LSHFunction
    similarity :: F

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
    function MonteCarloHash(similarity::F, discrete_hashfn::H, μ, volume,
                            p, n_samples) where {F, H<:LSHFunction}
        sample_points = [μ() for ii = 1:n_samples]

        T = eltype(μ())
        volume = T(volume)
        p = T(p)

        new{F,H,typeof(μ),T,eltype(sample_points)}(
            similarity, discrete_hashfn, μ, volume,
            p, n_samples, sample_points
        )
    end
end

### External MonteCarloHash constructors

# TODO: restrict similarities. E.g. Jaccard should not be an available similarity
MonteCarloHash(similarity, args...; kws...) =
    MonteCarloHash(SimilarityFunction(similarity), args...; kws...)

const _valid_MonteCarloHash_similarities = (
    # Function space similarities
    (L1, L2, cossim),  
    # Discrete-space similarities corresponding to function space similarities
    (ℓ1, ℓ2, cossim),  
    # Order of L^p space that the similarity applies to
    (1, 2, 2),
)

for (fn_space_simfn, simfn, p) in zip(_valid_MonteCarloHash_similarities...)
    quote
        # Add dispatch for case in which we specify the similarity function
        # to be $fn_space_simfn
        function MonteCarloHash(sim::SimilarityFunction{$fn_space_simfn}, μ, args...;
                                n_samples::Int64=1024, volume=1.0, kws...)

            discrete_hashfn = LSHFunction($simfn, args...; kws...)
            MonteCarloHash($fn_space_simfn, discrete_hashfn, μ, volume,
                           $p, n_samples)
        end
    end |> eval
end

# Implementation of MonteCarloHash for invalid similarity functions. Just throws
# an error.
# Necessary because otherwise the first MonteCarloHash constructor will go into
# infinite recursion if it receives an invalid similarity function.
function MonteCarloHash(sim::SimilarityFunction, args...; kws...)
    valid_sims = _valid_MonteCarloHash_similarities[1] .|>
                 string  |>
                 collect |>
                 sort!
    valid_sims = join(valid_sims, ", ", " or ")
    ErrorException("similarity must be $(valid_sims)") |> throw
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

similarity(hashfn::MonteCarloHash) = hashfn.similarity

n_hashes(hashfn::MonteCarloHash) =
    n_hashes(hashfn.discrete_hashfn)

# TODO: this may not be true
collision_probability(hashfn::MonteCarloHash, args...; kws...) =
    collision_probability(hashfn.discrete_hashfn, args...; kws...)

#========================
SymmetricLSHFunction API compliance
========================#

(hashfn::MonteCarloHash)(f) = index_hash(hashfn, f)

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
