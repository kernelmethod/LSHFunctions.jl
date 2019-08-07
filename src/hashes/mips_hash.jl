"""
Asymmetric LSH for approximate maximum inner product search. Ref:

	https://arxiv.org/abs/1405.5869
"""
struct MIPSHash{T} <: AsymmetricLSHFunction{T}
	coeff_A :: Matrix{T}
	coeff_B :: Matrix{T}
	denom :: T
	shift :: Vector{T}
	Qshift :: Vector{T}
	m :: Integer
end

function MIPSHash{T}(input_length::Integer, n_hashes::Integer, denom::Real, m::Integer = 3) where {T <: LSH_FAMILY_DTYPES}
	coeff_A = Matrix{T}(undef, n_hashes, input_length)
	coeff_B = Matrix{T}(undef, n_hashes, m)
	denom = T(denom)
	shift = Vector{T}(undef, n_hashes)
	Qshift = Vector{T}(undef, n_hashes)

	hashfn = MIPSHash{T}(coeff_A, coeff_B, denom, shift, Qshift, m)
	redraw!(hashfn)
	return hashfn
end

MIPSHash(args...; kws...) =
	MIPSHash{Float32}(args...; kws...)

#=
Function definitions for the two hash functions used by the approximate MIPS LSH,
h(P(x)) and h(Q(x)) (where h is an L^2 LSH function).
=#

# Helper functions
mat(x :: AbstractVector) = reshape(x, length(x), 1)
mat(x :: AbstractMatrix) = x

#=
h(P(x)) definitions
=#

function MIPSHash_P(h::MIPSHash{T}, x::AbstractArray) where {T}
	norms = col_norms(x)
	maxnorm = maximum(norms)
	maxnorm = maxnorm == 0 ? 1 : maxnorm	# To handle some edge cases
	BLAS.scal!(length(norms), 1/maxnorm, norms, 1)

	# First, perform a matvec on x and the first array of coefficients.
	# Note: aTx is an n_hashes × n_inputs array
	aTx = h.coeff_A * x .* (1/maxnorm) |> mat

	# Compute norms^2, norms^4, ... norms^(2^m).
	# Multiply these by the second array of coefficients and add them to aTx, so
	# that in totality we compute
	#
	# 		aTx = [coeff_A, coeff_B] * P(x)
	# 			= [coeff_A, coeff_B] * [x; norms^2; ...; norms^(2^m)]
	#
	# By making these computations in a somewhat roundabout way (rather than following
	# the formula above), we save a lot of memory by avoiding concatenations.
	# Note that m is typically small, so these iterations don't do much to harm performance
	for ii = 1:h.m
		norms .^= 2
		BLAS.ger!(T(1), h.coeff_B[:,ii], norms, aTx)
	end

	# Compute the remainder of the hash the same way we'd compute an L^p distance LSH.
	@. aTx = aTx / h.denom + h.shift

	return floor.(Int32, aTx)
end

MIPSHash_P(h :: MIPSHash{T}, x :: AbstractArray{<:Real}; kws...) where {T <: LSH_FAMILY_DTYPES} =
	MIPSHash_P(h, T.(x); kws...)

MIPSHash_P(h :: MIPSHash{T}, x :: AbstractArray{T}; kws...) where {T <: LSH_FAMILY_DTYPES} =
	invoke(MIPSHash_P, Tuple{MIPSHash{T}, AbstractArray}, h, x; kws...)

MIPSHash_P(h :: MIPSHash{T}, x :: AbstractVector{T}; kws...) where {T <: LSH_FAMILY_DTYPES} =
	invoke(MIPSHash_P, Tuple{MIPSHash{T}, AbstractArray}, h, x; kws...) |> vec

#=
h(Q(x)) definitions
=#
function MIPSHash_Q(h :: MIPSHash, x :: AbstractArray)
	# First, perform a matvec on x and the first array of coefficients.
	# Note: aTx is an n_hashes × n_inputs array
	aTx = h.coeff_A * x |> mat

	# Normalize the query vectors. We perform normalization after computing
	# aTx (rather than before) so that we don't have to allocate a new array
	# of size(x). Moreover, for large input vectors, the size of aTx is typically
	# much smaller than the size of x.
	f(x::T) where {T} = (x ≈ T(0) ? T(1) : x)
	norms = col_norms(x)
	map!(f, norms, norms)

	aTx .= aTx ./ norms'

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
	@. aTx = aTx / h.denom + h.Qshift

	return floor.(Int32, aTx)
end

MIPSHash_Q(h :: MIPSHash{T}, x :: AbstractArray{<:Real}) where {T <: LSH_FAMILY_DTYPES} =
	MIPSHash_Q(h, T.(x))

MIPSHash_Q(h :: MIPSHash{T}, x :: AbstractArray{T}) where {T <: LSH_FAMILY_DTYPES} =
	invoke(MIPSHash_Q, Tuple{MIPSHash{T}, AbstractArray}, h, x)

MIPSHash_Q(h :: MIPSHash{T}, x :: AbstractVector{T}) where {T <: LSH_FAMILY_DTYPES} =
	invoke(MIPSHash_Q, Tuple{MIPSHash{T}, AbstractArray}, h, x) |> vec

#=
LSHFunction and AsymmetricLSHFunction API compliance
=#
index_hash(h :: MIPSHash, x) = MIPSHash_P(h, x)
query_hash(h :: MIPSHash, x) = MIPSHash_Q(h, x)

n_hashes(h::MIPSHash) = length(h.shift)
hashtype(::MIPSHash) = Vector{Int32}

function redraw!(h::MIPSHash{T}) where T
	map!(_ -> randn(T), h.coeff_A, h.coeff_A)
	map!(_ -> randn(T), h.coeff_B, h.coeff_B)
	map!(_ -> rand(T), h.shift, h.shift)
	h.Qshift .= h.coeff_B * fill(T(1/2), h.m) ./ h.denom .+ h.shift
end
