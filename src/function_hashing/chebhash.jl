#================================================================

ChebHash for hashing the L^2([-1,1]) function space.

================================================================#

using FFTW

#========================
Global constants
========================#

const _DEFAULT_CHEBHASH_INTERVAL = @interval(-1 ≤ x ≤ 1)

#========================
Typedefs
========================#

# B = basis, which is a Symbol (e.g. :Chebyshev)
struct ChebHash{B, F<:SimilarityFunction, H<:LSHFunction, I<:RealInterval}
    # Discrete-space hash function used after extracting Chebyshev polynomial
    # coefficients from the input function.
    discrete_hashfn :: H

    # Interval over which all input functions are defined.
    interval :: I

    ### Internal ChebHash constructors
    function ChebHash{B,F}(
                hashfn::H,
                interval::I
            ) where {B, F<:SimilarityFunction, H<:LSHFunction, I<:RealInterval}

        new{B,F,H,I}(hashfn, interval)
    end
end

### External ChebHash constructors
const _valid_ChebHash_similarities = (
    # Function space similarities
    (L2, cossim),
    # Discrete-space similarities corresponding to function space similarities
    (ℓ2, cossim),
)

@doc """
    ChebHash(sim, args...; interval=$(_DEFAULT_CHEBHASH_INTERVAL), kws...)

Samples a hash function from an LSH family for the similarity `sim` defined over the function space ``L^p_{\\mu}(\\Omega)``. `sim` may be one of the following:
$(
join(
    ["- `" * sim * "`" for sim in (_valid_ChebHash_similarities[1] .|> 
                                   string |>
                                   collect |>
                                   sort!)
    ],
    "\n"
)
)

`ChebHash` works by approximating a function by Chebyshev polynomials. You can choose the degree of the approximation to trade between speed and generating desirable hash collision probabilities.

!!! info "ChebHash limitations"
    `ChebHash` can only hash function spaces of the form ``L^2([a,b])``, where ``[a,b]`` is an interval on the real line. For a more versatile option, checkout out [`MonteCarloHash`](@ref).

# Arguments
- `sim`: the similarity function you want to hash on.
- `args...`: arguments to pass on when building the `LSHFunction` instance underlying the returned `ChebHash` struct.
- `kws...`: keyword arguments to pass on when building the `LSHFunction` instance underlying the returned `ChebHash` struct.

# Examples
Create a hash function for cosine similarity for functions in ``L^2([-1,1])``:

```jldoctest; setup = :(using LSHFunctions)
julia> hashfn = ChebHash(cossim, 50; interval=@interval(-1 ≤ x ≤ 1));

julia> n_hashes(hashfn)
50

julia> similarity(hashfn) == cossim
true

julia> hashtype(hashfn)
$(cossim |> LSHFunction |> hashtype)
```

Create a hash function for ``L^2`` distance defined over ``L^2([0,2\\pi])``. Hash the functions `f(x) = cos(x)` and `f(x) = x/(2π)` using the returned `ChebHash`:

```jldoctest; setup = :(using LSHFunctions, Random; Random.seed!(0))
julia> hashfn = ChebHash(L2, 3; interval=@interval(0 ≤ x ≤ 2π));

julia> hashfn(cos)
3-element Array{Int32,1}:
  3
 -1
 -2

julia> hashfn(x -> x/(2π))
3-element Array{Int32,1}:
 0
 1
 0
```

See also: [`MonteCarloHash`](@ref)
"""
ChebHash(similarity, args...; kws...) =
    ChebHash(SimilarityFunction(similarity), args...; kws...)

for (fn_sim, discrete_sim) in zip(_valid_ChebHash_similarities...)
    quote
        # Add an implementation of ChebHash that dispatches on the similarity
        # function fn_sim
        function ChebHash(sim::SimilarityFunction{$fn_sim},
                          args...;
                          interval::RealInterval = _DEFAULT_CHEBHASH_INTERVAL,
                          kws...) where S

            discrete_hashfn = LSHFunction($discrete_sim, args...; kws...)
            ChebHash{:Chebyshev,typeof(sim)}(discrete_hashfn, interval)
        end
    end |> eval
end

# Implementation of ChebHash for invalid similarity functions. Just throws
# a error.
# Necessary because otherwise the first external ChebHash constructor
# will go into infinite recursion if it receives an invalid similarity
# function.
function ChebHash(sim::SimilarityFunction, args...; kws...)
    valid_sims = _valid_MonteCarloHash_similarities[1] .|>
                 string  |>
                 collect |>
                 sort!
    valid_sims = join(valid_sims, ", ", " or ")
    ErrorException("similarity must be $(valid_sims)") |> throw
end

#========================
Helper functions for ChebHash
========================#

# Perform an order-N Chebyshev discrete transform on samples of a function
# f(x) in order to approximate the coefficients for the degree-N Chebyshev
# polynomial of best fit for f(x).
function cheb_coefficients(f, N)
    # Sample f(x) at the non-uniformly spaced nodes x[1], ..., x[N], where
    #
    #       x[i] = cos((i-1)π / (N-1))
    #
    x = cos.(range(0, π, length=N))
    fx = f.(x)

    dct(fx) * √(1/N)
end

function get_cheb_coefficients(interval::RealInterval, f; n_coeffs::Integer=1024)
    f_ = squash_function(interval, f)
    coeff = cheb_coefficients(f_, n_coeffs)
    coeff .* √width(interval)
end

# Transform a function f ∈ L^2([a,b]) so that the coefficients of its Chebyshev
# polynomial create an approximate isomorphism between L^2([a,b]) and ℓ2(N).
function squash_function(interval::RealInterval{T}, f) where T
    lower::T, upper::T = interval.lower, interval.upper

    α = (upper - lower) / π
    β = lower

    x -> @. f(α * acos(x) + β)
end

#========================
LSHFunction API compliance
========================#

n_hashes(hashfn::ChebHash) =
    n_hashes(hashfn.discrete_hashfn)

similarity(::ChebHash{T,SimilarityFunction{F}}) where {T,F} = F

hashtype(hashfn::ChebHash) =
    hashtype(hashfn.discrete_hashfn)

# TODO: this may not be true
collision_probability(hashfn::ChebHash, args...; kws...) =
    collision_probability(hashfn.discrete_hashfn, args...; kws...)

#===============
Hash computation
===============#

function (hashfn::ChebHash{:Chebyshev})(f; kws...)
    coeff = get_cheb_coefficients(hashfn.interval, f; kws...)
    hashfn.discrete_hashfn(coeff)
end

function index_hash(hashfn::ChebHash{:Chebyshev}, f; kws...)
    coeff = get_cheb_coefficients(hashfn.interval, f; kws...)
    index_hash(hashfn.discrete_hashfn, coeff)
end

function query_hash(hashfn::ChebHash{:Chebyshev}, f)
    coeff = get_cheb_coefficients(hashfn.interval, f; kws...)
    query_hash(hashfn.discrete_hashfn, coeff)
end

#========================
ChebHash API
========================#

# Compute the similarity between f and g in the embedded space
function embedded_similarity(hashfn::ChebHash{:Chebyshev}, f, g; kws...)
    f_coeff = get_cheb_coefficients(hashfn.interval, f; kws...)
    g_coeff = get_cheb_coefficients(hashfn.interval, g; kws...)

    simfun = similarity(hashfn)

    Lf, Lg = length(f_coeff), length(g_coeff)
    @views simfun(f_coeff[1:min(Lf,Lg)], g_coeff[1:min(Lf,Lg)])
end
