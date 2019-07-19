#=
Definitions for symmetric LSH families.
=#

import LinearAlgebra: norm
import LinearAlgebra.BLAS: ger!

"""
Cosine similarity LSH function.
"""
struct CosSimHash{T, A <: Matrix{T}} <: SymmetricLSHFamily{T}
	coeff :: A
end

CosSimHash{T}(input_length :: Integer, n_hashes :: Integer) where {T} =
	CosSimHash(randn(T, n_hashes, input_length))

CosSimHash(args...; kws...) =
	CosSimHash{Float32}(args...; kws...)

(h::CosSimHash)(x::AbstractArray) =
	(h.coeff * x) .≥ 0

# Perform type conversion to hit BLAS when necessary
(h::CosSimHash{T})(x::AbstractArray{<:Real}) where {T <: LSH_FAMILY_DTYPES} =
	h(T.(x))

(h::CosSimHash{T})(x::AbstractArray{T}) where {T <: LSH_FAMILY_DTYPES} =
	invoke(h, Tuple{AbstractArray}, x)

"""
L^p distance LSH function.
"""
struct LpDistHash{T, A <: AbstractMatrix{T}} <: SymmetricLSHFamily{T}
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