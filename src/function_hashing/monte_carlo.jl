#================================================================

MonteCarloHash for hashing function spaces.

================================================================#

#========================
Global constants
========================#

const _MONTECARLOHASH_DEFAULT_N_SAMPLES = 1024

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
const _valid_MonteCarloHash_similarities = (
    # Function space similarities
    (L1, L2, cossim),  
    # Discrete-space similarities corresponding to function space similarities
    (ℓ1, ℓ2, cossim),  
    # Order of L^p space that the similarity applies to
    (1, 2, 2),
)

# TODO: restrict similarities. E.g. Jaccard should not be an available similarity
@doc """
    MonteCarloHash(sim, ω, args...; volume=1.0, n_samples=$(_MONTECARLOHASH_DEFAULT_N_SAMPLES), kws...)

Samples a hash function from an LSH family for the similarity `sim` defined over the function space ``L^p_{\\mu}(\\Omega)``. `sim` may be one of the following:
$(
join(
    ["- `" * sim * "`" for sim in (_valid_MonteCarloHash_similarities[1] .|> 
                                   string |>
                                   collect |>
                                   sort!)
    ],
    "\n"
)
)

Given an input function ``f\\in L^p_{\\mu}(\\Omega)``, `MonteCarloHash` works by sampling ``f`` at some randomly-selected points in ``\\Omega``, and then hashing those samples.

# Arguments
- `sim`: the similarity statistic you want to hash on.
- `ω`: a function that takes no inputs and samples a single point from ``\\Omega``. Alternatively, it can be viewed as a random variable with probability measure

```math
\\frac{\\mu}{\\text{vol}_{\\mu}(\\Omega)} = \\frac{\\mu}{\\int_{\\Omega} d\\mu}
```

- `args...`: arguments to pass on when building the `LSHFunction` instance underlying the returned `MonteCarloHash` struct.
- `volume::Real` (default: `1.0`): the volume of the space ``\\Omega``, defined as

```math
\\text{vol}_{\\mu}(\\Omega) = \\int_{\\Omega} d\\mu
```

- `n_samples::Integer` (default: `$(_MONTECARLOHASH_DEFAULT_N_SAMPLES)`): the number of points to sample from each function that is hashed by the `MonteCarloHash`. Larger values of `n_samples` tend to capture the input function better and will thus be more likely to achieve desirable collision probabilities.
- `kws...`: keyword arguments to pass on when building the `LSHFunction` instance underlying the returned `MonteCarloHash` struct.

# Examples
Create a hash function for cosine similarity for functions in ``L^2([-1,1])``:

```jldoctest; setup = :(using LSHFunctions)
julia> μ() = 2*rand()-1;   # μ samples a random point from [-1,1]

julia> hashfn = MonteCarloHash(cossim, μ, 50; volume=2.0);

julia> n_hashes(hashfn)
50

julia> similarity(hashfn) == cossim
true

julia> hashtype(hashfn)
$(cossim |> LSHFunction |> hashtype)
```

Create a hash function for ``L^2`` distance in the function space ``L^2([0,2\\pi])``. Hash the functions `f(x) = cos(x)` and `f(x) = x/(2π)` using the returned `MonteCarloHash`.

```jldoctest; setup = :(using LSHFunctions, Random; Random.seed!(0))
julia> μ() = 2π * rand(); # μ samples a random point from [0,2π]

julia> hashfn = MonteCarloHash(L2, μ, 3; volume=2π);

julia> hashfn(cos)
3-element Array{Int32,1}:
 -1
  3
  0

julia> hashfn(x -> x/(2π))
3-element Array{Int32,1}:
 -1
 -2
 -1
```

Create a hash function with a different number of sample points.

```jldoctest; setup = :(using LSHFunctions; μ() = rand())
julia> μ() = rand();  # Samples a random point from [0,1]

julia> hashfn = MonteCarloHash(cossim, μ; volume=1.0, n_samples=512);

julia> length(hashfn.sample_points)
512
```

See also: [`ChebHash`](@ref)
"""
MonteCarloHash(similarity, args...; kws...) =
    MonteCarloHash(SimilarityFunction(similarity), args...; kws...)

for (fn_space_simfn, simfn, p) in zip(_valid_MonteCarloHash_similarities...)
    quote
        # Add dispatch for case in which we specify the similarity function
        # to be $fn_space_simfn
        function MonteCarloHash(
                    sim::SimilarityFunction{$fn_space_simfn},
                    μ,
                    args...;
                    n_samples::Integer=_MONTECARLOHASH_DEFAULT_N_SAMPLES,
                    volume=1.0,
                    kws...)

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
