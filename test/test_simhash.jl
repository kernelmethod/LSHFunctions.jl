using Test, Random, LSH

@testset "SimHash tests" begin
	Random.seed!(0)
	import LSH: SymmetricLSHFunction

	@testset "Can construct a cosine similarity hash function" begin
		input_length = 5
		n_hashes = 8
		hashfn = SimHash(input_length, n_hashes)

		@test BitArray{1} == hashtype(hashfn)
	end

	@testset "Type consistency in SimHash fields" begin
		hashfn = SimHash{Float32}(1, 1)
		@test isa(hashfn, SimHash{Float32})
		@test isa(hashfn, SymmetricLSHFunction)
		@test isa(hashfn.coeff, Matrix{Float32})

		hashfn = SimHash{Float64}(1, 1)
		@test isa(hashfn, SimHash{Float64})
		@test isa(hashfn, SymmetricLSHFunction)
		@test isa(hashfn.coeff, Matrix{Float64})

		# The default should be for hashfn to be a SimHash{Float32}
		hashfn = SimHash(1, 1)
		@test isa(hashfn, SimHash{Float32})
		@test isa(hashfn.coeff, Matrix{Float32})
	end

	@testset "Hash simple inputs" begin
		input_length = 5
		n_hashes = 128
		hashfn = SimHash(input_length, n_hashes)

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

	@testset "SimHash collision probabilities match expectations" begin
	    # Test that the collision probability for SimHash is what's
	    # advertised by constructing a large number of hash functions
	    # and ensuring that the frequency of collisions is close
	    # to the computed probability.
	    input_length = 4
	    n_hashes = 1024
	    hashfn = SimHash{Float64}(input_length, n_hashes)

	    test_collision_probability(δ) = begin
	        x, y = rand(input_length), rand(input_length)
	        sim = CosSim(x,y)
	        prob = LSH.single_hash_collision_probability(hashfn, sim)

	        hx, hy = hashfn(x), hashfn(y)
            collision_frequency = mean(hx .== hy)

            # Check that the collision frequency is within ±δ of prob
            prob - δ ≤ collision_frequency ≤ prob + δ
	    end

	    @test test_collision_probability(0.05)
	    @test all(test_collision_probability(0.05) for ii = 1:128)
	end

	@testset "Hashing returns the correct data types" begin
		hashfn = SimHash(5, 2)

		## Test 1: Vector{Float64} -> BitArray{1}
		hashes = hashfn(randn(5))
		@test isa(hashes, BitArray{1})

		## Test 2: Matrix{Float64} -> BitArray{2}
		hashes = hashfn(randn(5, 10))
		@test isa(hashes, BitArray{2})
	end
end
