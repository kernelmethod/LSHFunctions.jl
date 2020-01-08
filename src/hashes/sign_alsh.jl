#================================================================

Definition of SignALSH, an LSH function for hashing on inner products.

================================================================#

import LinearAlgebra: norm

#========================
Typedefs
========================#

"""
Implementation of the SignALSH maximum inner product search (MIPS)
hash function. Ref:

    https://arxiv.org/abs/1410.5410v2
"""
mutable struct SignALSH{T <: Union{Float32,Float64}} <: AsymmetricLSHFunction
    coeff_A :: Matrix{T}
    coeff_B :: Matrix{T}
    P_shift :: Vector{T}
    m :: Int64

    # Whether or not SignALSH should round up to the next power of 2 when
    # resizing its coefficient array.
    resize_pow2 :: Bool
end

### External SignALSH constructors
function SignALSH{T}(n_hashes::Integer;
                     m::Integer = 3,
                     resize_pow2::Bool = false) where {T}

    coeff_A = Matrix{T}(undef, n_hashes, 0)
    coeff_B = randn(T, n_hashes, m)
    P_shift = coeff_B * fill(T(1/2), m)

    SignALSH(coeff_A, coeff_B, P_shift, Int64(m), resize_pow2)
end

SignALSH(args...; dtype=Float32, kws...) =
    SignALSH{dtype}(args...; kws...)

#============
SignALSH helper functions
=============#

function Base.resize!(hashfn::SignALSH{T}, n::Integer) where T
    n = (hashfn.resize_pow2) ? nextpow(2, n) : n

    # Note: the only field of SignALSH that is dependent on the input size is
    # coeff_A
    n_hashes, old_n = size(hashfn.coeff_A)
    old_coeff_A = hashfn.coeff_A
    new_coeff_A = similar(old_coeff_A, n_hashes, n)

    new_coeff_A[1:end,1:min(n,old_n)] .= old_coeff_A

    if n > old_n
        new_coeff_slice = @views new_coeff_A[1:end,old_n+1:end]
        @views map!(x -> randn(T), new_coeff_slice, new_coeff_slice)
    end

    hashfn.coeff_A = new_coeff_A
end

current_max_input_size(hashfn::SignALSH) =
    size(hashfn.coeff_A, 2)

#============
Hash computation implementation
=============#

#=
h(P(x)) definitions
=#
SignALSH_P(hashfn::SignALSH{T}, x::AbstractArray) where T =
    SignALSH_P(hashfn, T.(x))

function SignALSH_P(hashfn::SignALSH{T}, x::AbstractArray{T}) where {T}
    # SignALSH_P is essentially just SimHash on
    #
    #	P(x) = [x; 1/2-norms^2; 1/2-norms^4; ... 1/2-norms^(2^m)]
    #
    # after dividing through x by the largest norm of all of the columns
    # of x.
    norms = col_norms(x)
    maxnorm = maximum(norms)
    maxnorm = maxnorm == 0 ? 1 : maxnorm	# To handle some edge cases
    norms .*= 1/maxnorm

    n = size(x,1)
    if n > current_max_input_size(hashfn)
        resize!(hashfn, n)
    end

    Ax = @views hashfn.coeff_A[1:end,1:n] * x .* (1/maxnorm)

    # Perform the transformation P(x) on x, except that instead of actually
    # allocating memory for it and computing it, pile it onto Ax
    @. Ax += hashfn.P_shift

    for ii = 1:hashfn.m
        # Iteratively compute norms^2, norms^4, ... norms^(2^m). Multiply them by
        # columns of coeff_B and add to aTx, so that we end up with
        #
        # 	Ax = [coeff_A, coeff_B] * P(x)
        # 		= [coeff_A, coeff_B] * [x; 1/2-norms^2; ...; 1/2-norms^(2^m)]
        #
        # Note that we don't need to account for the 1/2 terms, which are
        # accounted for by the earlier addition of P_shift.
        norms .^= 2
        SignALSH_P_update_Ax!(hashfn.coeff_B[:,ii], norms, Ax)
    end

    return Ax .≥ 0
end

SignALSH_P_update_Ax!(coeff::Vector{T}, norms::Vector{T}, Ax::Array{T}) where T =
	BLAS.ger!(T(-1), coeff, norms, Ax)

# When the coefficients or norms are AbstractVectors, cascade through to reshape
# them into AbstractMatrix before updating Ax.
SignALSH_P_update_Ax!(coeff::AbstractVector, norms::AbstractVector, Ax) =
    SignALSH_P_update_Ax!(reshape(coeff, length(coeff), 1), norms, Ax)

SignALSH_P_update_Ax!(coeff, norms::AbstractVector, Ax) =
    SignALSH_P_update_Ax!(coeff, reshape(norms, length(norms), 1), Ax)

SignALSH_P_update_Ax!(coeff, norms, Ax) =
    (Ax .-= coeff * norms')

#=
h(Q(x)) definitions
=#

SignALSH_Q(hashfn::SignALSH{T}, x::AbstractArray) where T =
    SignALSH_Q(hashfn, T.(x))

SignALSH_Q(hashfn::SignALSH{T}, x::AbstractVector{T}) where T =
    invoke(SignALSH_Q, Tuple{SignALSH{T},AbstractArray{T}}, hashfn, x) |> vec

function SignALSH_Q(hashfn::SignALSH{T}, x::AbstractArray{T}) where {T}
    n = size(x,1)

    if n > current_max_input_size(hashfn)
        resize!(hashfn, n)
    end

    Ax = @views hashfn.coeff_A[1:end,1:n] * x
    norms = col_norms(x)
    map!(inv, norms, norms)
    @. Ax * norms' ≥ T(0)
end

#========================
LSHFunction and AsymmetricLSHFunction API compliance
========================#
index_hash(h::SignALSH, x) = SignALSH_P(h, x)
query_hash(h::SignALSH, x) = SignALSH_Q(h, x)

n_hashes(h::SignALSH) = size(h.coeff_A, 1)
hashtype(::SignALSH) = BitArray{1}
