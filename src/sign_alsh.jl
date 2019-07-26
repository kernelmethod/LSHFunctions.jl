import LinearAlgebra: norm

"""
Implementation of the SignALSH maximum inner product search (MIPS)
hash function. Ref:

	https://arxiv.org/abs/1410.5410v2
"""
struct SignALSH{T, A <: AbstractMatrix{T}} <: AsymmetricLSHFunction{T}
	coeff_A :: A
	coeff_B :: A
	P_shift :: Vector{T}
	m :: Int64
end

#=
Constructors
=#

function SignALSH{T}(input_length::Integer, n_hashes::Integer, m::Integer = 3) where {T}
	coeff_A = randn(T, n_hashes, input_length)
	coeff_B = randn(T, n_hashes, m)

	# Pre compute coeff_B * [1/2, 1/2, ..., 1/2] (as used in the transformation
	# P(x)) to reduce the cost of the indexing hash.
	P_shift = coeff_B * fill(T(1/2), m)

	SignALSH(coeff_A, coeff_B, P_shift, Int64(m))
end

SignALSH(args...; kws...) =
	SignALSH{Float32}(args...; kws...)

#=
Definitions of h(P(x)) for SignALSH
=#

# h(P(x)) definitions
function SignALSH_P(h::SignALSH{T}, x::AbstractArray) where {T}
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

	Ax = h.coeff_A * x .* (1/maxnorm)

	# Perform the transformation P(x) on x, except that instead of actually
	# allocating memory for it and computing it, pile it onto Ax
	@. Ax += h.P_shift

	for ii = 1:h.m
		# Iteratively compute norms^2, norms^4, ... norms^(2^m). Multiply them by
		# columns of coeff_B and add to aTx, so that we end up with
		#
		# 	Ax = [coeff_A, coeff_B] * P(x)
		# 		= [coeff_A, coeff_B] * [x; 1/2-norms^2; ...; 1/2-norms^(2^m)]
		#
		# Note that we don't need to account for the 1/2 terms, which are accounted
		# for by the earlier addition of P_shift.
		norms .^= 2
		BLAS.ger!(T(-1), h.coeff_B[:,ii], norms, Ax)
	end

	return Ax .≥ 0
end

SignALSH_P(h::SignALSH{T}, x::AbstractArray{<:Real}) where {T<:LSH_FAMILY_DTYPES} =
	SignALSH_P(h, T.(x))

SignALSH_P(h::SignALSH{T}, x::AbstractArray{T}) where {T<:LSH_FAMILY_DTYPES} =
	invoke(SignALSH_P, Tuple{SignALSH{T},AbstractArray}, h, x)

SignALSH_P(h::SignALSH{T}, x::AbstractVector{T}) where {T<:LSH_FAMILY_DTYPES} =
	invoke(SignALSH_P, Tuple{SignALSH{T},AbstractArray}, h, x) |> vec

# h(Q(x)) definitions
function SignALSH_Q(h::SignALSH{T}, x::AbstractArray) where {T}
	Ax = h.coeff_A * x
	norms = col_norms(x)
	map!(inv, norms, norms)
	@. Ax * norms' ≥ T(0)
end

SignALSH_Q(h::SignALSH{T}, x::AbstractArray{<:Real}) where {T<:LSH_FAMILY_DTYPES} =
	SignALSH_Q(h, T.(x))

SignALSH_Q(h::SignALSH{T}, x::AbstractArray{T}) where {T <: LSH_FAMILY_DTYPES} =
	invoke(SignALSH_Q, Tuple{SignALSH{T},AbstractArray}, h, x)

SignALSH_Q(h::SignALSH{T}, x::AbstractVector{T}) where {T <: LSH_FAMILY_DTYPES} =
	invoke(SignALSH_Q, Tuple{SignALSH{T},AbstractArray}, h, x) |> vec

#=
LSHFunction and AsymmetricLSHFunction API compliacne
=#
index_hash(h::SignALSH, x) = SignALSH_P(h, x)
query_hash(h::SignALSH, x) = SignALSH_Q(h, x)
hashtype(::SignALSH) = Bool
n_hashes(h::SignALSH) = size(h.coeff_A, 1)
