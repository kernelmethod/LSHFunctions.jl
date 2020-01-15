#================================================================

Common helper functions shared between multiple routines in the LSH module.

================================================================#

# Compute the norms of vectors and columns of matrices
col_norms(x::Union{AbstractVector,AbstractMatrix}) =
	map(norm, eachcol(x))

col_norms(x::Union{Vector,Matrix}) =
	map(BLAS.nrm2, eachcol(x))

col_norms(x::SparseVector) =
	[BLAS.nrm2(x.nzval)]

col_norms(x::SparseMatrixCSC{T}) where {T} = begin
	output = Vector{T}(undef, size(x,2))
	@inbounds for ii = 1:size(x,2)
		result = T(0)
		start_idx, end_idx = x.colptr[ii], x.colptr[ii+1]-1
		@simd for idx = start_idx:end_idx
			result += x.nzval[idx].^2
		end
		output[ii] = √result
	end
	return output
end
