"""
Cosine similarity LSH function.
"""
struct SimHash{T, A <: Matrix{T}} <: SymmetricLSHFunction{T}
	coeff :: A
end

SimHash{T}(input_length :: Integer, n_hashes :: Integer) where {T} =
	SimHash(randn(T, n_hashes, input_length))

SimHash(args...; kws...) =
	SimHash{Float32}(args...; kws...)

(h::SimHash)(x::AbstractArray) =
	(h.coeff * x) .≥ 0

# Perform type conversion to hit BLAS when necessary
(h::SimHash{T})(x::AbstractArray{<:Real}) where {T <: LSH_FAMILY_DTYPES} =
	h(T.(x))

(h::SimHash{T})(x::AbstractArray{T}) where {T <: LSH_FAMILY_DTYPES} =
	invoke(h, Tuple{AbstractArray}, x)

#=
LSHFunction and SymmetricLSHFunction API compliance
=#
hashtype(::SimHash) = Bool
