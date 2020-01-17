#================================================================

Definition of SimHash, an LSH function for hashing on cosine similarity.

================================================================#

#========================
Typedefs
========================#

mutable struct SimHash{T <: Union{Float32,Float64}} <: SymmetricLSHFunction
    # Random coefficients sampled for SimHash. Each column of coefficients
    # corresponds to another hash function.
    coeff :: Matrix{T}

    # Whether or not SimHash should round up to the next power of 2 when
    # resizing its coefficient array.
    #
    # This is useful if you're going to be hashing inputs of many different
    # sizes, since every call to resize! entails allocating a new array and
    # copying the old coefficient array into it. However, if you're only
    # going to be hashing inputs of one size or a few different sizes,
    # then you should set resize_pow2 to false.
    resize_pow2 :: Bool
end


function SimHash{T}(n_hashes::Integer = DEFAULT_N_HASHES;
                    resize_pow2::Bool = DEFAULT_RESIZE_POW2) where T

    coeff = Matrix{T}(undef, 0, n_hashes)
    SimHash{T}(coeff, resize_pow2)
end

SimHash(args...; dtype = DEFAULT_DTYPE, kws...) =
    SimHash{dtype}(args...; kws...)

@doc """
    SimHash(n_hashes::Integer = $(DEFAULT_N_HASHES);
            dtype::DataType = $(DEFAULT_DTYPE),
            resize_pow2::Bool = $(DEFAULT_RESIZE_POW2))

Creates a locality-sensitive hash function for cosine similarity.

# Arguments
- $(N_HASHES_DOCSTR())

# Keyword parameters
- $(DTYPE_DOCSTR(SimHash))
- $(RESIZE_POW2_DOCSTR(SimHash))

# Examples
Construct a hash function by calling `SimHash` with the number of hash functions you want to generate:

```jldoctest; setup = :(using LSH)
julia> hashfn = SimHash(24);

julia> n_hashes(hashfn) == 24 &&
       similarity(hashfn) == cossim
true
```

You can then call `hashfn(x)` in order to compute hashes:

```jldoctest; setup = :(using LSH)
julia> hashfn = SimHash(32);

julia> x = randn(30);

julia> hashes = hashfn(x);

```

# References

- Moses S. Charikar. *Similarity estimation techniques from rounding algorithms*. In Proceedings of the Thiry-Fourth Annual ACM Symposium on Theory of Computing, STOC '02, page 380–388, New York, NY, USA, 2002. Association for Computing Machinery. 10.1145/509907.509965.

See also: [`cossim`](@ref)
""" SimHash

#========================
SimHash helper functions
========================#

# Resize SimHash to hash vectors of length up to n
function Base.resize!(hashfn::SimHash, n :: Integer)
    # If resize_pow2 is set, then change n to the next power of 2 that
    # is larger than n.
    n = (hashfn.resize_pow2) ? nextpow(2, n) : n

    old_n, n_hashes = size(hashfn.coeff)
    old_coeff = hashfn.coeff
    new_coeff = similar(hashfn.coeff, n, n_hashes)

    # Copy old coefficients into the new_coeff Matrix
    @views new_coeff[1:min(n,old_n),1:end] .= old_coeff

    # Generate new coefficients in the empty entries
    if n > old_n
        # TODO: implement with map! to reduce memory allocation
        @views new_coeff[old_n+1:end,1:end] = randn(n - old_n, n_hashes)
    end

    hashfn.coeff = new_coeff
end

# The current maximum input size that a SimHash struct can accept based on
# the number of coefficients that we've generated for it. current_ is
# prepended because this implementation of SimHash supports vectors of
# different sizes.
current_max_input_size(hashfn::SimHash) = size(hashfn.coeff, 1)

#========================
LSHFunction and SymmetricLSHFunction API compliance
========================#

hashtype(::SimHash) = Bool
n_hashes(hashfn::SimHash) = size(hashfn.coeff, 2)
similarity(::SimHash) = cossim
single_hash_collision_probability(::SimHash, sim::Real) = (1 - acos(sim) / π)

### Hash computation

# Perform type conversion to hit BLAS when the input array has a different
# element type than the hashfn coefficients.
(hashfn::SimHash{T})(x::AbstractArray) where T =
	hashfn(T.(x))

function (hashfn::SimHash{T})(x::AbstractArray{T}) where T
    n = size(x,1)
    if n > current_max_input_size(hashfn)
        resize!(hashfn, n)
    end

    coeff = @views hashfn.coeff[1:n,1:end]

    (coeff' * x) .≥ 0
end

