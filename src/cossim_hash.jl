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

#=
LSHFamily and SymmetricLSHFamily API compliance
=#
hashtype(::CosSimHash) = Bool
