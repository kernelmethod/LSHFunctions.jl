#=
Locality-sensitive hashing type definitions and functions.
=#

import LinearAlgebra: norm
import LinearAlgebra.BLAS: ger!

const LSH_FAMILY_DTYPES = Union{Float32,Float64}
abstract type LSHFamily{T<:LSH_FAMILY_DTYPES} end

"""
Cosine similarity LSH function.
"""
struct CosSimHash{T, A <: Matrix{T}} <: LSHFamily{T}
	coeff :: A
end

CosSimHash{T}(input_length :: Integer, n_hashes :: Integer) where {T} =
	CosSimHash(randn(T, n_hashes, input_length))

CosSimHash(args...; kws...) =
	CosSimHash{Float32}(args...; kws...)

(h::CosSimHash)(x) = (h.coeff * x) .≥ 0

"""
L^p distance LSH function.
"""
struct LpDistHash{T, A <: AbstractMatrix{T}} <: LSHFamily{T}
	coeff :: A
	denom :: T
	shift :: Vector{T}
end

function LpDistHash{T}(input_length::Integer, n_hashes::Integer, denom::Real, power::Integer = 2) where {T}
	coeff = begin
		if power == 1
			rand(Cauchy(0, 1), n_hashes, input_length)
		elseif power == 2
			randn(n_hashes, input_length)
		end
	end

	LpDistHash{T}(T.(coeff), denom)
end

LpDistHash{T}(coeff :: A, denom :: Real) where {T, A <: AbstractMatrix{T}} =
	LpDistHash{T,A}(coeff, T(denom), rand(T, size(coeff, 1)))

LpDistHash(args...; kws...) =
	LpDistHash{Float32}(args...; kws...)

# L1DistHash and L2DistHash convenience wrappers
#
# NOTE: at the moment, it is impossible to pass type parameters to either of these
# wrappers. That means that users are stuck with the default type for LpDistHash
# structs if they use either of the following methods, instead of the general
# LpDistHash constructor.
L1DistHash(input_length :: Integer, n_hashes :: Integer, denom :: Real; kws...) where {T} =
	LpDistHash(input_length, n_hashes, denom, power = 1; kws...)

L2DistHash(input_length :: Integer, n_hashes :: Integer, denom :: Real; kws...) where {T} =
	LpDistHash(input_length, n_hashes, denom, power = 2; kws...)

# Definition of the actual hash function
function (h::LpDistHash)(x::AbstractArray)
	coeff, denom, shift = h.coeff, h.denom, h.shift
	hashes = coeff * x
	hashes = @. (hashes + shift) / denom
	floor.(Int32, hashes)
end

# When the input x does not already have the appropriate type, perform a type
# conversion first so that we can hit BLAS
(h::LpDistHash{T})(x::AbstractArray{<:Real}) where {T <: LSH_FAMILY_DTYPES} =
	h(T.(x))

(h::LpDistHash{T})(x::AbstractArray{T}) where {T <: LSH_FAMILY_DTYPES} =
	invoke(h, Tuple{AbstractArray}, x)

"""
Asymmetric LSH for approximate maximum inner product search. Ref:

	https://arxiv.org/abs/1405.5869
"""
struct MIPSHash{T} <: LSHFamily{T}
	coeff_A :: Matrix{T}
	coeff_B :: Matrix{T}
	denom :: T
	shift :: Vector{T}
	Qshift :: Vector{T}
	m :: Integer
end

function MIPSHash{T}(input_length::Integer, n_hashes::Integer, denom::Real, m::Integer) where {T <: LSH_FAMILY_DTYPES}
	coeff_A = randn(T, n_hashes, input_length)
	coeff_B = randn(T, n_hashes, m)
	denom = T(denom)
	shift = rand(T, n_hashes)
	Qshift = coeff_B * fill(T(1/2), m)

	MIPSHash{T}(coeff_A, coeff_B, denom, shift, Qshift, m)
end

MIPSHash(args...; kws...) =
	MIPSHash{Float32}(args...; kws...)

#=
Function definitions for the two hash functions used by the approximate MIPS LSH,
h(P(x)) and h(Q(x)) (where h is an L^2 LSH function).
=#
function MIPSHash_P_LSH(h :: MIPSHash{T}, x :: AbstractArray) where {T <: LSH_FAMILY_DTYPES}
	# First, perform a matvec on x and the first array of coefficients.
	# Note: aTx is an n_hashes × n_inputs array
	aTx = h.coeff_A * x

	# Compute the norms of the inputs, followed by norms^2, norms^4, ... norms^(2^m).
	# Multiply these by the second array of coefficients and add them to aTx, so
	# that in totality we compute
	#
	# 		aTx = [coeff_A, coeff_B] * P(x)
	# 			= [coeff_A, coeff_B] * [x; norms; norms^2; ...; norms^(2^m)]
	#
	# By making these computations in a somewhat roundabout way (rather than following
	# the formula above), we save a lot of memory by avoiding concatenations.
	norms = norm.(eachcol(x))
	ger!(T(1), h.coeff_B[:,ii], norms, aTx)

	# Note that m is typically small, so these iterations don't do much to harm performance
	for ii = 2:h.m
		@. norms = norms^2
		ger!(T(1), h.coeff_B[:,ii], norms, aTx)
	end

	# Compute the remainder of the hash the same way we'd compute an L^p distance LSH.
	@. aTx = (aTx + h.shift) / h.denom

	return floor.(Int32, aTx)
end

MIPSHash_P_LSH(h :: MIPSHash{T}, x :: AbstractArray{<:Real}) where {T <: LSH_FAMILY_DTYPES} =
	MIPSHash_P_LSH(h, T.(x))

MIPSHash_P_LSH(h :: MIPSHash{T}, x :: AbstractArray{T}) where {T <: LSH_FAMILY_DTYPES} =
	invoke(MIPSHash_P_LSH, Tuple{MIPSHash, AbstractArray}, h, x)

function MIPSHash_Q_LSH(h :: MIPSHash{T}, x :: AbstractArray) where {T <: LSH_FAMILY_DTYPES}
	# First, perform a matvec on x and the first array of coefficients.
	# Note: aTx is an n_hashes × n_inputs array
	aTx = h.coeff_A * x

	# Here, we would multiply the second array of coefficients by the elements that
	# Q(x) concatenates to x. Then we'd add this to aTx so that in total we compute
	#
	#		aTx = [coeff_A, coeff_B] * Q(x)
	#			= [coeff_A, coeff_B] * [x; 1/2; 1/2; ...; 1/2]
	#
	# Then we'd proceed with computing the rest of the L^2 distance LSH. However,
	# since the values concatenated on by Q(x) are always the same, we actually
	# pre-compute coeff_B * [1/2; 1/2; ...; 1/2] + shift when we construct the
	# MIPSHash to reduce the number of computations.
	@. aTx = (aTx + h.Qshift) / h.denom

	return floor.(Int32, aTx)
end

MIPSHash_Q_LSH(h :: MIPSHash{T}, x :: AbstractArray{<:Real}) where {T <: LSH_FAMILY_DTYPES} =
	MIPSHash_Q_LSH(h, T.(x))

MIPSHash_Q_LSH(h :: MIPSHash{T}, x :: AbstractArray{T}) where {T <: LSH_FAMILY_DTYPES} =
	invoke(MIPSHash_Q_LSH, Tuple{MIPSHash{T}, AbstractArray}, h, x)
