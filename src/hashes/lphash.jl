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

	# "Denominator" parameter r. Higher values of r lead to higher collision
	# rates. This parameter is user-specified
	r :: T

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

### External LpHash constructors

function LpHash{T}(n_hashes::Integer = 1;
                   r::Real = T(1.0),
                   power::Integer = 2,
                   resize_pow2::Bool = false) where {T <: Union{Float32,Float64}}

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

	LpHash(coeff, T(r), shift, Int64(power), distr, resize_pow2)
end

L1Hash(args...; kws...) where {T} = LpHash(args...; power = 1, kws...)

L2Hash(args...; kws...) where {T} = LpHash(args...; power = 2, kws...)

LpHash(args...; dtype::DataType = Float32, kws...) =
    LpHash{dtype}(args...; kws...)

# Documentation for L1Hash and L2Hash
@doc raw"""
    L1Hash(n_hashes::Integer = 1;
           dtype::DataType = Float32,
           r::Real = 1.0,
           resize_pow2::Bool = false)
    L2Hash(n_hashes::Integer = 1;
           dtype::DataType = Float32,
           r::Real = 1.0,
           resize_pow2::Bool = false)

Constructs a locality-sensitive hash for ``\ell^p`` distance (``\|x - y\|_p``). `L1Hash` constructs a hash function for ``\ell^1`` distance, and `L2Hash` constructs a hash function for ``\ell^2`` distance.

# Arguments
- `n_hashes::Integer` (default: `1`): the number of hash functions to generate.

# Keyword parameters
- `dtype::DataType` (default: `Float32`): the type to use for the resulting `LSH.LpHash`'s coefficients. Can be either `Float32` or `Float64`. You generally want to pick `dtype` to match the type of the data you're hashing.
- `r::Real` (default: `1.0`): a positive coefficient whose magnitude influences the collision rate. Larger values of `r` will increase the collision rate, even for distant points. See references for more information.
- `resize_pow2::Bool` (default: `false`): affects the way in which the `LSH.LpHash` struct resizes to hash inputs of different sizes. If you think you'll be hashing inputs of many different sizes, it's more efficient to set `resize_pow2 = true`.

# Examples
Construct an `LSH.LpHash` by calling `L1Hash` or `L2Hash` with the number of hash functions you want to generate:

```jldoctest; setup = :(using LSH)
julia> hashfn = L1Hash();

julia> hashfn.power == 1 &&
       n_hashes(hashfn) == 1 &&
       similarity(hashfn) == ℓ_1
true

julia> hashfn = L2Hash(128);

julia> hashfn.power == 2 &&
       n_hashes(hashfn) == 128 &&
       similarity(hashfn) == ℓ_2
true
```

After creating a hash function, you can compute hashes with `hashfn(x)`:

```jldoctest; setup = :(using LSH)
julia> hashfn = L1Hash(20);

julia> x = rand(4);

julia> hashes = hashfn(x);

```

# References

```
Datar, Mayur & Indyk, Piotr & Immorlica, Nicole & Mirrokni, Vahab. (2004). Locality-sensitive hashing scheme based on p-stable distributions. Proceedings of the Annual Symposium on Computational Geometry. 10.1145/997817.997857.
```

See also: [`ℓ_p`](@ref), [`ℓ_1`](@ref), [`ℓ_2`](@ref)
""" L1Hash

@doc (@doc L1Hash) L2Hash

#========================
Helper functions for LpHash
========================#

# Raise an error if the power corresponding to the L^p distance function is
# invalid
check_LpHash_power(hashfn::LpHash) = check_LpHash_power(hashfn.power)

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
hashtype(::LpHash) = Vector{Int32}

LSH.@register_similarity!(ℓ_1, L1Hash)
LSH.@register_similarity!(ℓ_2, L2Hash)

# See Section 3.2 of the reference
function single_hash_collision_probability(hashfn::LpHash, sim::Real)
    distr, r = hashfn.distr, hashfn.r
    integral, err = quadgk(x -> pdf(distr, x/sim) * (1 - x/r), 0, r, rtol=1e-5)
    integral /= sim

    # Note that from the reference for the L^p LSH family, we're supposed to integrate
    # over the PDF for the _absolute value_ of the underlying random variable, rather
    # than the raw PDF. Luckily, all of the distributions we have to deal with here
    # are symmetric and centered at zero, so all we have to do is multiply the
    # integral by two.
    integral *= 2
end

function similarity(hashfn::LpHash)
    if hashfn.power == 1
        ℓ_1
    elseif hashfn.power == 2
        ℓ_2
    else
        (x,y) -> ℓ_p(hashfn.power, x, y)
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
    h = @. h / hashfn.r + hashfn.shift
    floor.(Int32, h)
end
