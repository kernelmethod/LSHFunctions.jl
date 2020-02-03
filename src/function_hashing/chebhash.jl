#================================================================

ChebHash for hashing the L^2([-1,1]) function space.

================================================================#

using FFTW

#========================
Typedefs
========================#

# B = basis, which is a Symbol (e.g. :Chebyshev)
struct ChebHash{B, H<:LSHFunction, I<:RealInterval}
    discrete_hashfn :: H

    # Interval over which all input functions are defined.
    interval :: I
end

### External ChebHash constructors
ChebHash{S}(hashfn::H, interval::I) where {S, H<:LSHFunction, I<:RealInterval} =
    ChebHash{S,H,I}(hashfn, interval)

ChebHash(similarity, args...; kws...) =
    ChebHash(SimilarityFunction(similarity), args...; kws...)

function ChebHash(::SimilarityFunction{S},
                  args...;
                  interval::RealInterval = LSHFunctions.@interval(-1 ≤ x ≤ 1),
                  kws...) where S

    discrete_hashfn = LSHFunction(S, args...; kws...)
    ChebHash{:Chebyshev}(discrete_hashfn, interval)
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

    dct(fx) * √(1/(2N))
end

function get_cheb_coefficients(interval::RealInterval, f; n_coeffs::Integer=1024)
    f_ = squash_function(interval, f)
    coeff = cheb_coefficients(f_, n_coeffs)
    coeff .* width(interval)
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

similarity(hashfn::ChebHash) =
    similarity(hashfn.discrete_hashfn)

hashtype(hashfn::ChebHash) =
    hashtype(hashfn.discrete_hashfn)

# TODO: this may not be true
collision_probability(hashfn::ChebHash, args...; kws...) =
    collision_probability(hashfn.discrete_hashfn, args...; kws...)

#===============
Hash computation
===============#

function (hashfn::ChebHash{:Chebyshev})(f; kws...)
    coeff = get_cheb_coefficients(hashfn.interval, f, kws...)
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
function embedded_similarity(hashfn::ChebHash{:Chebyshev}, f, g)
    f_coeff = get_cheb_coefficients(hashfn.interval, f)
    g_coeff = get_cheb_coefficients(hashfn.interval, g)

    simfun = similarity(hashfn)

    Lf, Lg = length(f_coeff), length(g_coeff)
    @views simfun(f_coeff[1:min(Lf,Lg)], g_coeff[1:min(Lf,Lg)])
end
