import LinearAlgebra: norm
import LinearAlgebra.BLAS: ger!

"""
L^p distance LSH function.
"""
struct LpHash{T, A <: AbstractMatrix{T}} <: SymmetricLSHFunction{T}
	coeff :: A
	denom :: T
	shift :: Vector{T}
end

function LpHash{T}(input_length::Integer, n_hashes::Integer, denom::Real, power::Integer = 2) where {T}
	coeff = begin
		if power == 1
			rand(Cauchy(0, 1), n_hashes, input_length)
		elseif power == 2
			randn(n_hashes, input_length)
		end
	end

	LpHash{T}(T.(coeff), denom)
end

LpHash{T}(coeff :: A, denom :: Real) where {T, A <: AbstractMatrix{T}} =
	LpHash{T,A}(coeff, T(denom), rand(T, size(coeff, 1)))

LpHash(args...; kws...) =
	LpHash{Float32}(args...; kws...)

# L1Hash and L2Hash convenience wrappers
#
# NOTE: at the moment, it is impossible to pass type parameters to either of these
# wrappers. That means that users are stuck with the default type for LpHash
# structs if they use either of the following methods, instead of the general
# LpHash constructor.
L1Hash(input_length :: Integer, n_hashes :: Integer, denom :: Real; kws...) where {T} =
	LpHash(input_length, n_hashes, denom, power = 1; kws...)

L2Hash(input_length :: Integer, n_hashes :: Integer, denom :: Real; kws...) where {T} =
	LpHash(input_length, n_hashes, denom, power = 2; kws...)

# Definition of the actual hash function
function (h::LpHash)(x::AbstractArray)
	coeff, denom, shift = h.coeff, h.denom, h.shift
	hashes = coeff * x
	hashes = @. hashes / denom + shift
	floor.(Int32, hashes)
end

# When the input x does not already have the appropriate type, perform a type
# conversion first so that we can hit BLAS
(h::LpHash{T})(x::AbstractArray{<:Real}) where {T <: LSH_FAMILY_DTYPES} =
	h(T.(x))

(h::LpHash{T})(x::AbstractArray{T}) where {T <: LSH_FAMILY_DTYPES} =
	invoke(h, Tuple{AbstractArray}, x)

#=
LSHFunction and SymmetricLSHFunction API compliance
=#
hashtype(::LpHash) = Int32
