using Test, Random, LSH, LinearAlgebra, SparseArrays

include(joinpath("..", "test_utils.jl"))

#==================
Tests
==================#
@testset "SignALSH tests" begin
	Random.seed!(RANDOM_SEED)

	@test_skip @testset "Can hash inputs correctly with SignALSH" begin
		input_length = 5
		n_hashes = 8

		hashfn = SignALSH(input_length, n_hashes, 3)
		simhash = SimHash([hashfn.coeff_A hashfn.coeff_B])
		
		# By default, hashfn should be using Float32 for its coefficients
		@test isa(hashfn, SignALSH{Float32})

		X = rand(Float32, input_length, 128)
		ihashes = index_hash(hashfn, X)
		qhashes = query_hash(hashfn, X)

		@test eltype(ihashes) == eltype(qhashes)
		@test BitArray{1} == hashtype(hashfn)

		# 1. Compute the indexing hashes manually
		norms = map(norm, eachcol(X))
		maxnorm = maximum(norms)
		norms .*= 1 / maxnorm
		Y = X ./ maxnorm

		@test all(map(norm,eachcol(Y)) .≤ 1)
		@test all(0 .≤ norms .≤ 1)

		norm_powers = -[norms.^2 norms.^4 norms.^8] .+ 1/2
		Px = [Y; norm_powers']

		@test simhash(Px) == ihashes

		# 2. Compute the querying hashes manually
		Qx = [X ./ norms'; zeros(1,128); zeros(1,128); zeros(1,128)]

		@test simhash(Qx) == qhashes
	end

	@testset "SignALSH generates collisions for large inner products" begin
		input_length = 5; n_hashes = 128;
		hashfn = SignALSH(input_length, n_hashes)

		x = randn(input_length)
		x_query_hashes = query_hash(hashfn, x)

		# Check that SignALSH isn't just generating a single query hash
		@test any(x_query_hashes .!= x_query_hashes[1])

		# Compute the indexing hashes for a dataset with four vectors:
		# a) 10 * x (where x is the test query vector)
		# b) x
		# c) A vector of all zeros
		# d) -x
		dataset = [(10*x) x zeros(input_length) -x]
		p_hashes = index_hash(hashfn, dataset)

		# Each collection of hashes should be different from one another
		@test let result = true
			for (ii,jj) in product(1:4, 1:4)
				if ii != jj && p_hashes[:,ii] == p_hashes[:,jj]
					result = false
					break
				end
			end
			result
		end

		# The number of collisions should be highest for x and 2*x, second-highest
		# for x and x, second-lowest for x and zeros, and lowest for x and -x
		n_collisions = [sum(x_query_hashes .== p) for p in eachcol(p_hashes)]
		@test n_collisions[1] > n_collisions[2] > n_collisions[3] > n_collisions[4]
	end

	@testset "Can hash sparse arrays" begin
		input_size = 100
		n_inputs = 150
		n_hashes = 2

		hashfn = SignALSH(input_size, n_hashes)
		x = sprandn(input_size, n_inputs, 0.2)

		# Mostly just need to test that the following lines don't crash
		ih = index_hash(hashfn, x)
		qh = query_hash(hashfn, x)

		@test size(ih) == (n_hashes, n_inputs)
		@test size(qh) == (n_hashes, n_inputs)
	end

	@testset "Can hash matrix adjoints" begin
		input_size = 100
		n_inputs = 150
		n_hashes = 2
		hashfn = SignALSH(input_size, n_hashes)

		## Test 1: regular matrix adjoint
		x = randn(n_inputs, input_size)'
		@test index_hash(hashfn, x) == index_hash(hashfn, copy(x))
		@test query_hash(hashfn, x) == query_hash(hashfn, copy(x))

		## Test 2: sparse matrix adjoint
		x = sprandn(n_inputs, input_size, 0.2)'
		@test index_hash(hashfn, x) == index_hash(hashfn, copy(x))
		@test query_hash(hashfn, x) == query_hash(hashfn, copy(x))
	end
end
