#================================================================

Definition of MIPSHash for hashing on inner products.

================================================================#

#========================
Typedefs
========================#

"""
Asymmetric LSH for approximate maximum inner product search. Ref:

	https://arxiv.org/abs/1405.5869
"""
struct MIPSHash{T <: Union{Float32,Float64}} <: AsymmetricLSHFunction
    coeff_A :: Matrix{T}
    coeff_B :: Matrix{T}
    denom :: T
    shift :: Vector{T}
    Qshift :: Vector{T}
    m :: Integer
end

### External MIPSHash constructors
function MIPSHash{T}(
        input_length::Integer,
        n_hashes::Integer,
        denom::Real,
        m::Integer = 3) where {T <: Union{Float32,Float64}}

    coeff_A = randn(T, n_hashes, input_length)
    coeff_B = randn(T, n_hashes, m)
    denom = T(denom)
    shift = rand(T, n_hashes)
    Qshift = coeff_B * fill(T(1/2), m) ./ denom .+ shift

	MIPSHash{T}(coeff_A, coeff_B, denom, shift, Qshift, m)
end

MIPSHash(args...; kws...) =
	MIPSHash{Float32}(args...; kws...)

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

function _MIPSHash_P(h :: MIPSHash{T}, x :: AbstractArray) where {T}
    norms = col_norms(x)
    maxnorm = maximum(norms)
    maxnorm = maxnorm == 0 ? 1 : maxnorm	# To handle some edge cases
    BLAS.scal!(length(norms), 1/maxnorm, norms, 1)

    # First, perform a matvec on x and the first array of coefficients.
    # Note: aTx is an n_hashes × n_inputs array
    aTx = h.coeff_A * x .* (1/maxnorm) |> mat

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
    for ii = 1:h.m
        norms .^= 2
        MIPSHash_P_update_aTx!(h.coeff_B[:,ii], norms, aTx)
    end

    # Compute the remainder of the hash the same way we'd compute an L^p distance LSH.
    @. aTx = aTx / h.denom + h.shift

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

function _MIPSHash_Q(hashfn::MIPSHash, x::AbstractArray)
    # First, perform a matvec on x and the first array of coefficients.
    # Note: aTx is an n_hashes × n_inputs array
    aTx = hashfn.coeff_A * x |> mat

    # Normalize the query vectors. We perform normalization after computing
    # aTx (rather than before) so that we don't have to allocate a new array
    # of size(x). Moreover, for large input vectors, the size of aTx is typically
    # much smaller than the size of x.
    f(x::T) where {T} = (x ≈ T(0) ? T(1) : x)
    norms = col_norms(x)
    map!(f, norms, norms)

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
    @. aTx = aTx / hashfn.denom + hashfn.Qshift

    return floor.(Int32, aTx)
end

#========================
LSHFunction and AsymmetricLSHFunction API compliance
========================#
index_hash(hashfn::MIPSHash, x) = MIPSHash_P(hashfn, x)
query_hash(hashfn::MIPSHash, x) = MIPSHash_Q(hashfn, x)

n_hashes(hashfn::MIPSHash) = length(hashfn.shift)
hashtype(::MIPSHash) = Vector{Int32}
