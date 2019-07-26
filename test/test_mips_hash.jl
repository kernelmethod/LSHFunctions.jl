using Test, Random, LSH

@testset "MIPSHash tests" begin
	Random.seed!(0)
	import LinearAlgebra: norm
	import Base.Iterators: product
	import SparseArrays: sprandn

	@testset "Can construct a simple MIPS hash function" begin
		input_length = 5
		n_hashes = 8
		denom = 2
		m = 3

		hashfn = MIPSHash(input_length, n_hashes, denom, m)
		@test size(hashfn.coeff_A) == (n_hashes, input_length)
		@test size(hashfn.coeff_B) == (n_hashes, m)
		@test size(hashfn.shift) == (n_hashes,)
		@test size(hashfn.Qshift) == (n_hashes,)
		@test hashfn.denom == denom
		@test hashfn.m == m

		# The default datatype should be Float32
		@test isa(hashfn, MIPSHash{Float32})
	end

	@testset "Type consistency in MIPSHash fields" begin
		# Check for type consistency between fields of the struct, so that we
		# avoid expensive type conversions during runtime.
		for T in (Float32, Float64)
			hashfn = MIPSHash{T}(5, 5, 2, 4)

			@test isa(hashfn.coeff_A, Matrix{T})
			@test isa(hashfn.coeff_B, Matrix{T})
			@test isa(hashfn.shift, Vector{T})
			@test isa(hashfn.Qshift, Vector{T})
			@test isa(hashfn.denom, T)
		end
	end

	@testset "Hashing returns the correct data types" begin
		hashfn = MIPSHash{Float64}(4, 1, 1., 1)

		# Matrix{Float64} -> Matrix{Int32}
		x = randn(4, 10)
		p_hashes = index_hash(hashfn, x)
		q_hashes = query_hash(hashfn, x)

		@test isa(p_hashes, Matrix{Int32})
		@test isa(q_hashes, Matrix{Int32})
		@test eltype(p_hashes) == hashtype(hashfn)
		@test eltype(q_hashes) == hashtype(hashfn)

		# Vector{Float64} -> Vector{Int32}
		x = randn(4)
		p_hashes = index_hash(hashfn, x)
		q_hashes = query_hash(hashfn, x)

		@test isa(index_hash(hashfn, x), Vector{Int32})
		@test isa(query_hash(hashfn, x), Vector{Int32})
		@test eltype(p_hashes) == hashtype(hashfn)
		@test eltype(q_hashes) == hashtype(hashfn)
	end

	@testset "MIPSHash h(P(x)) is correctly computed" begin
		input_length = 5; n_hashes = 128; denom = 0.5
		hashfn = MIPSHash(input_length, n_hashes, denom, 3)
		coeff = [hashfn.coeff_A hashfn.coeff_B]
		shift = hashfn.shift

		@test size(coeff) == (n_hashes, input_length+3)
		@test size(shift) == (n_hashes,)

		## Test 1: compute hashes on a single input
		x = randn(input_length)
		hash = index_hash(hashfn, x)

		@test isa(hash, Vector{Int32})
		@test length(hash) == n_hashes

		# Start by performing the transform P(x)
		u = x / norm(x)
		norm_powers = [norm(u)^2, norm(u)^4, norm(u)^8]
		Px = [u; norm_powers]

		# Now compute the L^2 hash of P(x)
		manual_hash = coeff * Px ./ denom .+ shift
		manual_hash = floor.(Int32, manual_hash)

		@test manual_hash == hash

		## Test 2: compute hashes on many inputs simultaneously
		n_inputs = 256
		x = randn(input_length, n_inputs)
		hashes = index_hash(hashfn, x)

		@test isa(hashes, Matrix{Int32})
		@test size(hashes) == (n_hashes, n_inputs)

		# Scale the inputs so that they each have norm ≤ 1
		norms = norm.(eachcol(x))
		max_norm = maximum(norms)
		u = x ./ max_norm

		# Now re-compute the norms and their first few powers, and
		# append to the matrix u.
		norms = norm.(eachcol(u))
		norm_powers = [norms.^2 norms.^4 norms.^8]
		Px = [u; norm_powers']

		# Now compute the L^2 hash of Px
		manual_hashes = coeff * Px ./ denom .+ shift
		manual_hashes = floor.(Int32, manual_hashes)

		@test manual_hashes == hashes
	end

	@testset "MIPSHash h(Q(x)) is correctly computed" begin
		input_length = 5; n_hashes = 128; denom = 0.5
		hashfn = MIPSHash(input_length, n_hashes, denom, 3)
		coeff = [hashfn.coeff_A hashfn.coeff_B]
		shift = hashfn.shift

		@test size(coeff) == (n_hashes, input_length+3)
		@test size(shift) == (n_hashes,)

		## Test 1: test on a single input
		x = randn(input_length)
		hash = query_hash(hashfn, x)

		@test isa(hash, Vector{Int32})
		@test length(hash) == n_hashes
		
		# To compute the hash manually, we start by creating the
		# transform Q(x)
		u = x ./ norm(x)
		Qx = [u; 1/2; 1/2; 1/2]

		@test size(Qx) == (input_length+3,)

		# Then, we compute the L^2 hash of Qx
		manual_hash = coeff * Qx ./ denom .+ shift
		manual_hash = floor.(Int32, manual_hash)
		
		@test manual_hash == hash

		## Test 2: test on multiple inputs
		n_inputs = 256
		x = randn(input_length, n_inputs)
		hashes = query_hash(hashfn, x)

		@test isa(hashes, Matrix{Int32})
		@test size(hashes) == (n_hashes, n_inputs)

		u = x ./ norm.(eachcol(x))'
		Qx = [u; fill(1/2, 3, n_inputs)]

		manual_hashes = coeff * Qx ./ denom .+ shift
		manual_hashes = floor.(Int32, manual_hashes)

		@test manual_hashes == hashes
	end

	@testset "MIPSHash generates collisions for large inner products" begin
		input_length = 5; n_hashes = 128; denom = 1; m = 5
		hashfn = MIPSHash(input_length, n_hashes, denom, m)

		x = randn(input_length)
		x_query_hashes = query_hash(hashfn, x)

		# Check that MIPSHash isn't just generating a single query hash
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

	@testset "Can compute hashes for sparse arrays" begin
		X = sprandn(Float32, 10, 1000, 0.2)
		hashfn = MIPSHash(size(X,1), 8, 1, 1)

		ihashes = index_hash(hashfn, X)
		qhashes = query_hash(hashfn, X)

		# Compare against the case where X is dense
		X = Matrix(X)
		ihashes_dense = index_hash(hashfn, X)
		qhashes_dense = query_hash(hashfn, X)
		
		@test ihashes == ihashes_dense
		@test qhashes == qhashes_dense
	end
end
