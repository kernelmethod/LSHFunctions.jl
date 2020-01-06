#================================================================

Definition of SimHash, an LSH function for hashing on cosine similarity.

================================================================#

#========================
Typedefs
========================#

"""
Cosine similarity LSH function.
"""
struct SimHash{T <: Union{Float32,Float64}} <: SymmetricLSHFunction
	coeff :: Matrix{T}
end

function SimHash{T}(
        input_length :: Integer,
        n_hashes :: Integer) where {T <: Union{Float32,Float64}}

    coeff = randn(T, input_length, n_hashes)
    hashfn = SimHash(coeff)
end

SimHash(args...; kws...) =
	SimHash{Float32}(args...; kws...)

#========================
LSHFunction and SymmetricLSHFunction API compliance
========================#

n_hashes(h :: SimHash) = size(h.coeff, 2)
hashtype(:: SimHash) = BitArray{1}

single_hash_collision_probability(::SimHash, similarity) =
    (1 - acos(similarity) / π)

### Hash computation

(hashfn::SimHash{T})(x::AbstractArray{Real}) where T =
	hashfn(T.(x))

(hashfn::SimHash{T})(x::AbstractArray{T}) where T =
    (hashfn.coeff' * x) .≥ 0

