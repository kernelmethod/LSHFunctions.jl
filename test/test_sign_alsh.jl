using Test, Random, LSH, LinearAlgebra

@testset "SignALSH tests" begin
	Random.seed!(0)

	@testset "Can hash inputs correctly with SignALSH" begin
		input_length = 5
		n_hashes = 8

		hashfn = SignALSH(input_length, n_hashes, 3)
		simhash = SimHash([hashfn.coeff_A hashfn.coeff_B])
		
		# By default, hashfn should be using Float32 for its coefficients
		@test isa(hashfn, SignALSH{Float32})

		X = rand(Float32, input_length, 128)
		ihashes = index_hash(hashfn, X)
		qhashes = query_hash(hashfn, X)

		@test eltype(ihashes) == eltype(qhashes) == hashtype(hashfn)

		# 1. Compute the indexing hashes manually
		norms = map(norm, eachcol(X))
		maxnorm = maximum(norms)
		norm_powers = -[norms.^2 norms.^4 norms.^8] .+ 1/2
		Px = [X ./ maxnorm; norm_powers']

		@test simhash(Px) == ihashes

		# 2. Compute the query hashes manually
		Qx = [X ./ norms'; zeros(1,128); zeros(1,128); zeros(1,128)]

		@test simhash(Qx) == qhashes
	end
end
