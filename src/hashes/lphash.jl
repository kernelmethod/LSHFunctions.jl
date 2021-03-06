#================================================================

Definition of LpHash, an LSH function for hashing on L^p distance.

Primary reference:

    Datar, Mayur & Indyk, Piotr & Immorlica, Nicole & Mirrokni, Vahab. (2004).
    Locality-sensitive hashing scheme based on p-stable distributions.
    Proceedings of the Annual Symposium on Computational Geometry.
    10.1145/997817.997857.

================================================================#

using Distributions
using QuadGK: quadgk
import LinearAlgebra: norm
import LinearAlgebra.BLAS: ger!

#========================
Typedefs
========================#

mutable struct LpHash{T <: Union{Float32,Float64}, D} <: SymmetricLSHFunction
    # Coefficient matrix with which we multiply the input to the hash function
    coeff :: Matrix{T}

    # "Denominator" parameter scale (called "r" in the reference paper). Higher
    # values of r lead to higher collision rates. This parameter is
    # user-specified.
    scale :: T

    # "Shift" parameter (referred to as 'b' in the 'p-stable distributions' paper.
    # There's one shift parameter for every hash function; each parameter is
    # randomly sampled from [0,1].
    shift :: Vector{T}

    # The parameter p specifying the order of L^p distance we're using.
    power :: Int64

    # Distribution from which new coefficients are sampled.
    distr :: D

    # Whether or not LpHash should round up to the next power of 2 when
    # resizing its coefficient array.
    #
    # This is useful if you're going to be hashing inputs of many different
    # sizes, since every call to resize! entails allocating a new array and
    # copying the old coefficient array into it. However, if you're only
    # going to be hashing inputs of one size or a few different sizes,
    # then you should set resize_pow2 to false.
    resize_pow2 :: Bool
end

abstract type L1Hash <: SymmetricLSHFunction end
abstract type L2Hash <: SymmetricLSHFunction end

### External LpHash constructors

function LpHash{T}(n_hashes::Integer = DEFAULT_N_HASHES;
                   scale::Real = T(1.0),
                   power::Integer = 2,
                   resize_pow2::Bool = DEFAULT_RESIZE_POW2) where {T <: Union{Float32,Float64}}

    coeff = Matrix{T}(undef, n_hashes, 0)
    shift = rand(T, n_hashes)

    distr = begin
        if power == 1
            Cauchy(0,1)
        elseif power == 2
            Normal(0,1)
        else
            "power must be 1 or 2" |> ErrorException |> throw
        end
    end

    LpHash(coeff, T(scale), shift, Int64(power), distr, resize_pow2)
end

L1Hash(args...; kws...) where {T} = LpHash(args...; power = 1, kws...)

L2Hash(args...; kws...) where {T} = LpHash(args...; power = 2, kws...)

LpHash(args...; dtype::Type = DEFAULT_DTYPE, kws...) =
    LpHash{dtype}(args...; kws...)

### Documentation for L1Hash and L2Hash
for (hashfn, power) in zip((:L1Hash, :L2Hash), (1, 2))
    sim = "ℓ$(power)"
    equation = (power == 1) ?
                "\\|x - y\\|_$(power) = \\sum_i |x_i - y_i|" :
                "\\|x - y\\|_$(power) = \\left(\\sum_i |x_i - y_i|^$(power)\\right)^{1/$(power)}"

    quote
@doc """
    $($hashfn)(
        n_hashes::Integer = $(DEFAULT_N_HASHES);
        dtype::Type = $(DEFAULT_DTYPE),
        r::Real = 1.0,
        resize_pow2::Bool = $(DEFAULT_RESIZE_POW2)
    )

Constructs a locality-sensitive hash for ``\\ell^$($power)`` distance (``\\|x - y\\|_$($power)``), defined as

``$($equation)``

# Arguments
- $(N_HASHES_DOCSTR())

# Keyword parameters
- $(DTYPE_DOCSTR($hashfn))
- `r::Real` (default: `1.0`): a positive coefficient whose magnitude influences the collision rate. Larger values of `r` will increase the collision rate, even for distant points. See references for more information.
- $(RESIZE_POW2_DOCSTR($hashfn))

# Examples
Construct an `$($hashfn)` with the number of hash functions you want to generate:

```jldoctest; setup = :(using LSHFunctions)
julia> hashfn = $($hashfn)(128);

julia> hashfn.power == $($power) &&
       n_hashes(hashfn) == 128 &&
       similarity(hashfn) == $($sim)
true
```

After creating a hash function, you can compute hashes with `hashfn(x)`:

```jldoctest; setup = :(using LSHFunctions)
julia> hashfn = $($hashfn)(20);

julia> x = rand(4);

julia> hashes = hashfn(x);

```

# References

- Datar, Mayur & Indyk, Piotr & Immorlica, Nicole & Mirrokni, Vahab. (2004). *Locality-sensitive hashing scheme based on p-stable distributions*. Proceedings of the Annual Symposium on Computational Geometry. 10.1145/997817.997857.

See also: [`$($sim)`](@ref ℓp)
""" $hashfn
    end |> eval
end

#========================
Helper functions for LpHash
========================#

# Resize the coefficients of an LpHash to support vectors of size up to n
function Base.resize!(hashfn::LpHash{T}, n::Integer) where T
    # If resize_pow2 is set, then change n to the next power of n larger than n
    n = (hashfn.resize_pow2) ? nextpow(2, n) : n

    n_hashes, old_n = size(hashfn.coeff)
    old_coeff = hashfn.coeff
    new_coeff = similar(hashfn.coeff, n_hashes, n)

    # Copy old coefficients into the new_coeff Matrix
    @views new_coeff[1:end,1:min(n,old_n)] .= old_coeff

    # Generate new coefficients in the empty entries
    if n > old_n
        new_coeff_slice = @views new_coeff[1:end,old_n+1:end]
        distr = hashfn.distr
        map!(x -> T(rand(distr)), new_coeff_slice, new_coeff_slice)
    end

    hashfn.coeff = new_coeff
end

current_max_input_size(hashfn::LpHash) =
    size(hashfn.coeff, 2)

#========================
LSHFunction and SymmetricLSHFunction API compliance
========================#

n_hashes(h::LpHash) = length(h.shift)
hashtype(::LpHash) = Int32

# See Section 3.2 of the reference paper
function single_hash_collision_probability(hashfn::LpHash, sim::T) where {T <: Real}
    ### If sim ≈ 0 then the integral won't be possible to numerically compute,
    ### however we know that the probability equals one.
    if sim ≈ T(0)
        return T(1)
    end

    ### Compute the collision probability for a single hash function
    distr, scale = hashfn.distr, hashfn.scale
    integral, err = quadgk(x -> pdf(distr, x/sim) * (T(1) - x/scale),
                           T(0), T(scale), rtol=1e-5)
    integral = integral ./ sim

    # Note that from the reference for the L^p LSH family, we're supposed to
    # integrate over the p.d.f. for the _absolute value_ of the underlying
    # random variable, rather than the raw p.d.f. Luckily, all of the
    # distributions we have to deal with here are symmetric and centered at
    # zero, so all we have to do is multiply the integral by two.
    single_hash_prob = T(integral .* 2)
end

function similarity(hashfn::LpHash)
    if hashfn.power == 1
        ℓ1
    elseif hashfn.power == 2
        ℓ2
    else
        (x,y) -> ℓp(hashfn.power, x, y)
    end
end

### Hash computation

# Perform type conversion to hit BLAS
(hashfn::LpHash{T})(x::AbstractArray) where T =
    hashfn(T.(x)) 

function (hashfn::LpHash{T})(x::AbstractArray{T}) where T
    # Resize the coefficient array if necessary
    n = size(x,1)
    if n > current_max_input_size(hashfn)
        resize!(hashfn, n)
    end

    h = @views hashfn.coeff[1:end,1:n] * x
    h = @. h / hashfn.scale + hashfn.shift
    floor.(Int32, h)
end
