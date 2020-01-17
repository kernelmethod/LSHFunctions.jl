#================================================================

Definition of MIPSHash for hashing on inner products.

================================================================#

#========================
Typedefs
========================#

mutable struct MIPSHash{T <: Union{Float32,Float64}} <: AsymmetricLSHFunction
    coeff_A :: Matrix{T}
    coeff_B :: Matrix{T}
    scale :: T
    shift :: Vector{T}
    Qshift :: Vector{T}
    m :: Int64

    # An upper bound on the norm of the data points this hash function will
    # process
    maxnorm :: T

    # Whether or not the number of coefficients per hash function should be
    # expanded to be a power of 2 whenever we need to resize coeff_A.
    resize_pow2 :: Bool
end

### External MIPSHash constructors

@doc """
    MIPSHash(n_hashes::Integer = $(DEFAULT_N_HASHES);
             dtype::Datatype = $(DEFAULT_DTYPE),
             maxnorm::Union{Nothing,Real} = nothing,
             scale::Real = 1,
             m::Integer = 3,
             resize_pow2::Bool = $(DEFAULT_RESIZE_POW2))

Create a `MIPSHash` hash function for hashing on inner product similarity.

# Arguments
- $(N_HASHES_DOCSTR())

# Keyword parameters
- $(DTYPE_DOCSTR(MIPSHash))
- `maxnorm::Union{Nothing,Real}` (default: `nothing`): an upper bound on the ``\\ell^2``-norm of the data points. **Note: this keyword argument must be explicitly specified.** If it left unspecified (or set to `nothing`), `MIPSHash()` will raise an error.
- `scale::Real` (default: `1`): parameter that affects the probability of a hash collision. Large values of `scale` increases hash collision probability (even for inputs with low inner product similarity); small values of `scale` will decrease hash collision probability.

# Examples
`MIPSHash` is an [`AsymmetricLSHFunction`](@ref), and hence hashes must be computed using `index_hash` and `query_hash`.

```jldoctest; setup = :(using LSH)
julia> hashfn = MIPSHash(5; maxnorm=10);

julia> x = rand(4);

julia> ih = index_hash(hashfn, x); qh = query_hash(hashfn, x);

julia> length(ih) == length(qh) == 5
true

julia> typeof(ih) == typeof(qh) == Vector{Int32}
true
```

You need to explicitly specify the `maxnorm` keyword parameter when constructing `MIPSHash`, otherwise you will get an error.

```jldoctest; setup = :(using LSH)
julia> hashfn = MIPSHash(5)
ERROR: maxnorm must be specified for MIPSHash
```

You'll also get an error if you try to hash a vector that has norm greater than the `maxnorm` that you specified.

```jldoctest; setup = :(using LSH)
julia> hashfn = MIPSHash(; maxnorm=1);

julia> index_hash(hashfn, ones(4))
ERROR: norm 2.0 exceeds maxnorm (1.0)
```

# References
- Anshumali Shrivastava and Ping Li. *Asymmetric LSH (ALSH) for Sublinear Time Maximum Inner Product Search (MIPS)*. Proceedings of the 27th International Conference on Neural Information Processing Systems - Volume 2, NIPS'14, page 2321–2329, Cambridge, MA, USA, 2014. MIT Press. 10.5555/2969033.2969086. [https://arxiv.org/abs/1405.5869]

See also: [`inner_prod`](@ref), [`ℓ2_norm`](@ref ℓp_norm)
"""
@generated function MIPSHash{T}(n_hashes::Integer = DEFAULT_N_HASHES;
                                maxnorm::Union{Nothing,Real} = nothing,
                                scale::Real = 1,
                                m::Integer = 3,
                                resize_pow2::Bool = DEFAULT_RESIZE_POW2) where T
    if maxnorm <: Nothing
        :("maxnorm must be specified for MIPSHash" |>
          ErrorException |>
          throw)
    else
        quote
            if n_hashes < 1
                "n_hashes must be positive" |>
                ErrorException |>
                throw
            elseif scale ≤ 0
                "scaling factor `scale` must be positive" |>
                ErrorException |>
                throw
            elseif m ≤ 0
                "m must be positive" |>
                ErrorException |>
                throw
            elseif maxnorm ≤ 0
                "maxnorm must be positive" |>
                ErrorException |>
                throw
            end

            coeff_A = Matrix{T}(undef, n_hashes, 0)
            coeff_B = randn(T, n_hashes, m)
            scale = T(scale)
            m = Int64(m)
            shift = rand(T, n_hashes)
            Qshift = coeff_B * fill(T(1/2), m) ./ scale .+ shift

            MIPSHash{T}(coeff_A, coeff_B, scale, shift, Qshift, m,
	                    maxnorm, resize_pow2)
	    end
	end
end

MIPSHash(args...; dtype=DEFAULT_DTYPE, kws...) =
	MIPSHash{dtype}(args...; kws...)

#============
MIPSHash helper functions
=============#

function Base.resize!(hashfn::MIPSHash{T}, n::Integer) where T
    n = (hashfn.resize_pow2) ? nextpow(2, n) : n

    # The only field of MIPSHash that's dependent on the input size is coeff_A,
    # so we only need to resize that array.
    n_hashes, old_n = size(hashfn.coeff_A)
    old_coeff_A = hashfn.coeff_A
    new_coeff_A = similar(old_coeff_A, n_hashes, n)

    new_coeff_A[1:end, 1:min(n,old_n)] .= old_coeff_A

    if n > old_n
        new_coeff_slice = @views new_coeff_A[1:end,old_n+1:end]
        @views map!(x -> randn(T), new_coeff_slice, new_coeff_slice)
    end

    hashfn.coeff_A = new_coeff_A
end

current_max_input_size(hashfn::MIPSHash) = size(hashfn.coeff_A, 2)

#========================
Function definitions for the two hash functions used by the approximate MIPS LSH,
h(P(x)) and h(Q(x)) (where h is an L^2 LSH function).
========================#

# Helper functions
mat(x::AbstractVector) = reshape(x, length(x), 1)
mat(x::AbstractMatrix) = x

#=========
h(P(x)) definitions
=========#

@generated function MIPSHash_P(
        hashfn::MIPSHash{T},
        x::AbstractArray{S}) where {T,S}

    if T != S
        # Perform type conversion to hit BLAS
        :( MIPSHash_P(hashfn, T.(x)) )
    elseif x <: AbstractVector
        :( _MIPSHash_P(hashfn, x) |> vec )
    else
        :( _MIPSHash_P(hashfn, x) )
    end
end

function _MIPSHash_P(hashfn::MIPSHash{T}, x::AbstractArray) where {T}
    n = size(x,1)
    if n > current_max_input_size(hashfn)
        resize!(hashfn, size(x,1))
    end

    norms = col_norms(x)
    for norm_ii in norms
        if norm_ii > hashfn.maxnorm
            "norm $(norm_ii) exceeds maxnorm ($(hashfn.maxnorm))" |>
            ErrorException |>
            throw
        end
    end
    BLAS.scal!(length(norms), 1/hashfn.maxnorm, norms, 1)

    # First, perform a matvec on x and the first array of coefficients.
    # Note: aTx is an n_hashes × n_inputs array
    @views aTx = hashfn.coeff_A[1:end,1:n] * x .* (1/hashfn.maxnorm) |> mat

    # Compute norms^2, norms^4, ... norms^(2^m).
    # Multiply these by the second array of coefficients and add them to aTx, so
    # that in totality we compute
    #
    # 		aTx = [coeff_A, coeff_B] * P(x)
    # 			= [coeff_A, coeff_B] * [x; norms^2; ...; norms^(2^m)]
    #
    # By making these computations in a somewhat roundabout way (rather than
    # following the formula above), we save a lot of memory by avoiding
    # concatenations.
    # Note that m is typically small, so these iterations don't do much to harm
    # performance
    for ii = 1:hashfn.m
        norms .^= 2
        MIPSHash_P_update_aTx!(hashfn.coeff_B[:,ii], norms, aTx)
    end

    # Compute the remainder of the hash the same way we'd compute an L^p distance
    # LSH.
    @. aTx = aTx / hashfn.scale + hashfn.shift

    return floor.(Int32, aTx)
end

MIPSHash_P_update_aTx!(coeff::Vector{T}, norms::Vector{T}, aTx :: Array{T}) where T =
	BLAS.ger!(T(1), coeff, norms, aTx)

MIPSHash_P_update_aTx!(coeff, norms, aTx) =
	(aTx .+= coeff' * norms)

#==========
h(Q(x)) definitions
===========#

@generated function MIPSHash_Q(
        hashfn::MIPSHash{T},
        x::AbstractArray{S}) where {T,S}

    if T != S
        # Perform type conversion to hit BLAS
        :( MIPSHash_Q(hashfn, T.(x)) )
    elseif x <: AbstractVector
        :( _MIPSHash_Q(hashfn, x) |> vec )
    else
        :( _MIPSHash_Q(hashfn, x) )
    end
end

function _MIPSHash_Q(hashfn::MIPSHash{T}, x::AbstractArray) where T
    n = size(x,1)
    if n > current_max_input_size(hashfn)
        resize!(hashfn, n)
    end

    # First, perform a matvec on x and the first array of coefficients.
    # Note: aTx is an n_hashes × n_inputs array
    aTx = @views hashfn.coeff_A[1:end,1:n] * x |> mat

    # Normalize the query vectors. We perform normalization after computing
    # aTx (rather than before) so that we don't have to allocate a new array
    # of size(x). Moreover, for large input vectors, the size of aTx is typically
    # much smaller than the size of x.
    norms = col_norms(x)
    for norm_ii in norms
        if norm_ii > hashfn.maxnorm
            "norm $(norm_ii) exceeds maxnorm ($(hashfn.maxnorm))" |>
            ErrorException |>
            throw
        end
    end
    map!(x::T -> x ≈ T(0) ? T(1) : x, norms, norms)
    aTx .= aTx ./ norms'

    # Here, we would multiply the second array of coefficients by the elements
    # that Q(x) concatenates to x. Then we'd add this to aTx so that in total we
    # compute
    #
    #		aTx = [coeff_A, coeff_B] * Q(x)
    #			= [coeff_A, coeff_B] * [x; 1/2; 1/2; ...; 1/2]
    #
    # Then we'd proceed with computing the rest of the L^2 distance LSH. However,
    # since the values concatenated on by Q(x) are always the same, we actually
    # pre-compute coeff_B * [1/2; 1/2; ...; 1/2] + shift when we construct the
    # MIPSHash to reduce the number of computations.
    @. aTx = aTx / hashfn.scale + hashfn.Qshift

    return floor.(Int32, aTx)
end

#========================
LSHFunction and AsymmetricLSHFunction API compliance
========================#
index_hash(hashfn::MIPSHash, x) = MIPSHash_P(hashfn, x)
query_hash(hashfn::MIPSHash, x) = MIPSHash_Q(hashfn, x)
similarity(::MIPSHash) = inner_prod

n_hashes(hashfn::MIPSHash) = length(hashfn.shift)
hashtype(::MIPSHash) = Int32
