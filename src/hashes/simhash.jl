"""
Cosine similarity LSH function.
"""
struct SimHash{T, A <: AbstractMatrix{T}} <: SymmetricLSHFunction{T}
	coeff :: A
end

function SimHash{T}(input_length :: Integer, n_hashes :: Integer) where {T}
	coeff = Matrix{T}(undef, n_hashes, input_length)
	hashfn = SimHash(coeff)
	redraw!(hashfn)
end

SimHash(args...; kws...) =
	SimHash{Float32}(args...; kws...)

(h :: SimHash)(x :: AbstractArray) =
	(h.coeff * x) .≥ 0

# Perform type conversion to hit BLAS when necessary
(h :: SimHash{T})(x :: AbstractArray{<:Real}) where {T <: LSH_FAMILY_DTYPES} =
	h(T.(x))

(h :: SimHash{T})(x :: AbstractArray{T}) where {T <: LSH_FAMILY_DTYPES} =
	invoke(h, Tuple{AbstractArray}, x)

#=
LSHFunction and SymmetricLSHFunction API compliance
=#
n_hashes(h :: SimHash) = size(h.coeff, 1)
hashtype(:: SimHash) = BitArray{1}

function redraw!(h :: SimHash{T}) where {T}
	redraw!(h.coeff, () -> randn(T))
	return h
end
