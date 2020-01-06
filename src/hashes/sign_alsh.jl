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
struct SignALSH{T <: Union{Float32,Float64}} <: AsymmetricLSHFunction

    coeff_A :: Matrix{T}
    coeff_B :: Matrix{T}
    P_shift :: Vector{T}
    m :: Int64
end

### External SignALSH constructors
function SignALSH{T}(
        input_length::Integer,
        n_hashes::Integer,
        m::Integer = 3) where {T}

    coeff_A = randn(T, n_hashes, input_length)
    coeff_B = randn(T, n_hashes, m)
    P_shift = coeff_B * fill(T(1/2), m)

    SignALSH(coeff_A, coeff_B, P_shift, Int64(m))
end

SignALSH(args...; kws...) =
	SignALSH{Float32}(args...; kws...)

#============
Hash computation implementation
=============#

#=
h(P(x)) definitions
=#
SignALSH_P(h :: SignALSH{T}, x :: AbstractArray{<:Real}) where {T<:Union{Float32,Float64}} =
	SignALSH_P(h, T.(x))

SignALSH_P(h :: SignALSH{T}, x :: AbstractArray{T}) where {T<:Union{Float32,Float64}} =
	invoke(SignALSH_P, Tuple{SignALSH{T},AbstractArray}, h, x)

function SignALSH_P(h :: SignALSH{T}, x :: AbstractArray) where {T}
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
		SignALSH_P_update_Ax!(h.coeff_B[:,ii], norms, Ax)
	end

	return Ax .≥ 0
end

SignALSH_P_update_Ax!(coeff :: Vector{T}, norms :: Vector{T}, Ax :: Array{T}) where T =
	BLAS.ger!(T(-1), coeff, norms, Ax)

# When the coefficients or norms are AbstractVectors, cascade through to reshape
# them into AbstractMatrix before updating Ax.
SignALSH_P_update_Ax!(coeff :: AbstractVector, norms :: AbstractVector, Ax) =
	SignALSH_P_update_Ax!(reshape(coeff, length(coeff), 1), norms, Ax)

SignALSH_P_update_Ax!(coeff, norms :: AbstractVector, Ax) =
	SignALSH_P_update_Ax!(coeff, reshape(norms, length(norms), 1), Ax)

SignALSH_P_update_Ax!(coeff, norms, Ax) =
	(Ax .-= coeff * norms')

#=
h(Q(x)) definitions
=#
function SignALSH_Q(h :: SignALSH{T}, x :: AbstractArray) where {T}
	Ax = h.coeff_A * x
	norms = col_norms(x)
	map!(inv, norms, norms)
	@. Ax * norms' ≥ T(0)
end

SignALSH_Q(h :: SignALSH{T}, x :: AbstractArray{<:Real}) where {T<:Union{Float32,Float64}} =
	SignALSH_Q(h, T.(x))

SignALSH_Q(h :: SignALSH{T}, x :: AbstractArray{T}) where {T <: Union{Float32,Float64}} =
	invoke(SignALSH_Q, Tuple{SignALSH{T},AbstractArray}, h, x)

SignALSH_Q(h :: SignALSH{T}, x :: AbstractVector{T}) where {T <: Union{Float32,Float64}} =
	invoke(SignALSH_Q, Tuple{SignALSH{T},AbstractArray}, h, x) |> vec

#========================
LSHFunction and AsymmetricLSHFunction API compliance
========================#
index_hash(h :: SignALSH, x) = SignALSH_P(h, x)
query_hash(h :: SignALSH, x) = SignALSH_Q(h, x)

n_hashes(h :: SignALSH) = size(h.coeff_A, 1)
hashtype(:: SignALSH) = BitArray{1}

function redraw!(h :: SignALSH{T}) where {T}
end
