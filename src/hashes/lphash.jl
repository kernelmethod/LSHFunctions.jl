#================================================================

Definition of LpHash, an LSH function for hashing on L^p distance.

================================================================#

using Distributions
import LinearAlgebra: norm
import LinearAlgebra.BLAS: ger!

#========================
Typedefs
========================#

"""
L^p distance LSH function.
"""
struct LpHash{T <: Union{Float32,Float64}} <: SymmetricLSHFunction
	coeff :: Matrix{T}
	denom :: T
	shift :: Vector{T}
	power :: Int64
end

function LpHash{T}(
        input_length::Integer,
        n_hashes::Integer,
        denom::Real,
        power::Integer = 2) where {T <: Union{Float32,Float64}}

    # Draw random coefficients
    distr = begin
        if power == 1
            Cauchy(0,1)
        elseif power == 2
            Normal(0,1)
        else
            "power must be 1 or 2" |> ErrorException |> throw
        end
    end

    coeff = T.(rand(distr, n_hashes, input_length))
    shift = rand(T, n_hashes)

	LpHash{T}(coeff, T(denom), shift, Int64(power))
end

LpHash(args...; kws...) =
	LpHash{Float32}(args...; kws...)

# L1Hash and L2Hash convenience wrappers
#
# NOTE: at the moment, it is impossible to pass type parameters to either of these
# wrappers. That means that users are stuck with the default type for LpHash
# structs if they use either of the following methods, instead of the general
# LpHash constructor.
L1Hash(input_length::Integer, n_hashes::Integer, denom::Real; kws...) where {T} =
	LpHash(input_length, n_hashes, denom, power = 1; kws...)

L2Hash(input_length::Integer, n_hashes::Integer, denom::Real; kws...) where {T} =
	LpHash(input_length, n_hashes, denom, power = 2; kws...)

#========================
LSHFunction and SymmetricLSHFunction API compliance
========================#

n_hashes(h :: LpHash) = length(h.shift)
hashtype(:: LpHash) = Vector{Int32}

### Hash computation

@generated function (hashfn::LpHash{T})(x::AbstractArray{S}) where {T,S<:Real}
    if T == S
        quote
            h = hashfn.coeff * x
            h = @. h / hashfn.denom + hashfn.shift
            floor.(Int32, h)
        end
    else
        # If the elements of x don't already have type T, perform a type
        # conversion to hit BLAS
        :(hashfn(T.(x)))
    end
end
