using Test, Random, LSH

@testset "Cosine similarity tests" begin
	Random.seed!(0)
	import LSH: SymmetricLSHFamily

	@testset "Can construct a cosine similarity hash function" begin
		input_length = 5
		n_hashes = 8
		hashfn = CosSimHash(input_length, n_hashes)

		@test size(hashfn.coeff) == (n_hashes, input_length)

		# Test with just one hash
		hashfn = CosSimHash(input_length, 1)
		@test size(hashfn.coeff) == (1, input_length)
	end

	@testset "Type consistency in CosSimHash fields" begin
		hashfn = CosSimHash{Float32}(1, 1)
		@test isa(hashfn, CosSimHash{Float32})
		@test isa(hashfn, SymmetricLSHFamily{Float32})
		@test isa(hashfn.coeff, Matrix{Float32})

		hashfn = CosSimHash{Float64}(1, 1)
		@test isa(hashfn, CosSimHash{Float64})
		@test isa(hashfn, SymmetricLSHFamily{Float64})
		@test isa(hashfn.coeff, Matrix{Float64})

		# The default should be for hashfn to be a CosSimHash{Float32}
		hashfn = CosSimHash(1, 1)
		@test isa(hashfn, CosSimHash{Float32})
		@test isa(hashfn.coeff, Matrix{Float32})
	end

	@testset "Hash simple inputs" begin
		input_length = 5
		n_hashes = 128
		hashfn = CosSimHash(input_length, n_hashes)

		## Test 1: a single input that is just the zero vector
		x = zeros(input_length)
		@test all(hashfn(x) .== true)

		## Test 2: many inputs, all of which are zero vectors
		x = zeros(input_length, 32)
		@test all(hashfn(x) .== true)

		## Test 3: many identical inputs
		u = randn(input_length)
		x = similar(u, input_length, 32)
		x .= u
		hashes = hashfn(x)
		@test all(hashes[:,1] .== hashes)

		# For any input, the probability that all of its hashes are the
		# same should be extremely low, 2^(-n_hashes+1). Thus, there
		# should be at least one true bit and one false bit among
		# the hashes for an input.
		@test any(hashes[:,1]) && !all(hashes[:,1])
	end

	@testset "CosSimHash computes hashes correctly" begin
		input_length = 5
		n_hashes = 128
		hashfn = CosSimHash(input_length, n_hashes)
		coeff = hashfn.coeff

		## Test 1: a single input
		x = randn(input_length)
		hashes = hashfn(x)
		manual_hashes = coeff * x .≥ 0

		@test hashes == manual_hashes

		## Test 2: multiple inputs
		x = randn(input_length, 32)
		hashes = hashfn(x)
		manual_hashes = coeff * x .≥ 0

		@test hashes == manual_hashes
	end

	@testset "Hashing returns the correct data types" begin
		hashfn = CosSimHash(5, 2)

		## Test 1: Vector{Float64} -> BitArray{1}
		hashes = hashfn(randn(5))
		@test eltype(hashes) == hashtype(hashfn)
		@test isa(hashes, BitArray{1})

		## Test 2: Matrix{Float64} -> BitArray{2}
		hashes = hashfn(randn(5, 10))
		@test eltype(hashes) == hashtype(hashfn)
		@test isa(hashes, BitArray{2})
	end
end
