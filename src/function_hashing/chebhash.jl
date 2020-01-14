#================================================================

ChebHash for hashing the L^2([-1,1]) function space.

================================================================#

using ApproxFun: Chebyshev, Fun

#========================
Typedefs
========================#

struct ChebHash{H<:LSHFunction, I<:RealInterval}
    discrete_hashfn :: H

    # Interval over which all input functions are defined.
    interval :: I
end

### External ChebHash constructors
ChebHash(similarity, args...; kws...) =
    ChebHash(SimilarityFunction(similarity), args...; kws...)

function ChebHash(::SimilarityFunction{S},
                  args...;
                  interval::RealInterval = LSH.@interval(-1 ≤ x ≤ 1),
                  kws...) where S

    discrete_hashfn = LSHFunction(S, args...; kws...)
    ChebHash(discrete_hashfn, interval)
end

#========================
Helper functions for ChebHash
========================#

function get_cheb_coefficients(interval::RealInterval, f)
    f_ = squash_function(interval, f)
    cheb = Fun(f_, Chebyshev())
    cheb.coefficients .* (width(interval) / 2.0)
end

# Squash a function f into the interval [-1,1]
function squash_function(interval::RealInterval, f)
    lower, upper = interval.lower, interval.upper
    α = (upper - lower) / 2
    β = (upper + lower) / 2
    x -> @. √(1-x^2) * f(α*x + β)
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

#===============
Hash computation
===============#

function (hashfn::ChebHash)(f)
    coeff = get_cheb_coefficients(hashfn.interval, f)
    hashfn.discrete_hashfn(coeff)
end

function index_hash(hashfn::ChebHash, f)
    coeff = get_cheb_coefficients(hashfn.interval, f)
    index_hash(hashfn.discrete_hashfn, coeff)
end

function query_hash(hashfn::ChebHash, f)
    coeff = get_cheb_coefficients(hashfn.interval, f)
    query_hash(hashfn.discrete_hashfn, coeff)
end

#========================
ChebHash API
========================#

# Compute the similarity between f and g in the embedded space
function embedded_similarity(hashfn::ChebHash, f, g)
    f_coeff = get_cheb_coefficients(hashfn.interval, f)
    g_coeff = get_cheb_coefficients(hashfn.interval, g)

    simfun = similarity(hashfn)

    Lf, Lg = length(f_coeff), length(g_coeff)
    @views simfun(f_coeff[1:min(Lf,Lg)], g_coeff[1:min(Lf,Lg)])
end
