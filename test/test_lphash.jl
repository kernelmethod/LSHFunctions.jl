using Test, Random, LSH

@testset "LpHash tests" begin
	Random.seed!(0)
	import LSH: SymmetricLSHFunction

	@testset "Can construct a L^p distance hash function" begin
		input_length = 5
		n_hashes = 8
		denom = 2

		for p in (1,2)
			Lp_hash = LpHash(input_length, n_hashes, denom, p)

			@test size(Lp_hash.coeff) == (n_hashes, input_length)
			@test Lp_hash.denom == denom
			@test size(Lp_hash.shift) == (n_hashes,)
		end
	end

	@testset "Type consistency in LpHash fields" begin
		# Should have type consistency between the fields of the struct,
		# so that we avoid expensive type conversions.
		Lp_hash = LpHash{Float32}(5, 5, 1)

		@test isa(Lp_hash, LpHash{Float32})
		@test isa(Lp_hash.coeff, Matrix{Float32})
		@test isa(Lp_hash.denom, Float32)
		@test isa(Lp_hash.shift, Vector{Float32})

		Lp_hash = LpHash{Float64}(5, 5, 1)
		@test isa(Lp_hash, LpHash{Float64})
		@test isa(Lp_hash.coeff, Matrix{Float64})
		@test isa(Lp_hash.denom, Float64)
		@test isa(Lp_hash.shift, Vector{Float64})

		# The default dtype should be Float32
		Lp_hash = LpHash(5, 5, 1)
		@test isa(Lp_hash, LpHash{Float32})
		@test isa(Lp_hash.coeff, Matrix{Float32})
		@test isa(Lp_hash.denom, Float32)
		@test isa(Lp_hash.shift, Vector{Float32})
	end

	@testset "Hashes are correctly computed" begin
		input_length = 5
		n_hashes = 8
		denom = 2

		hashfn = LpHash{Float32}(input_length, n_hashes, denom)
		coeff, shift = hashfn.coeff, hashfn.shift

		# Test on a single input
		x = randn(input_length)
		hashes = hashfn(x)
		manual_hashes = floor.(Int32, coeff * x ./ denom .+ shift)

		@test isa(hashes, Vector{Int32})
		@test hashes == manual_hashes

		# Test on many inputs, simultaneously
		x = randn(input_length, 128)
		hashes = hashfn(x)
		manual_hashes = floor.(Int32, coeff * x ./ denom .+ shift)

		@test isa(hashes, Matrix{Int32})
		@test hashes == manual_hashes
	end

	@testset "Hashes have the correct dtype" begin
		hashfn = LpHash(5, 5, 1)

		# Test 1: Vector{Float64} -> Vector{Int32}
		hashes = hashfn(randn(5))
		@test Vector{eltype(hashes)} == hashtype(hashfn)
		@test isa(hashes, Vector{Int32})

		# Test 2: Matrix{Float64} -> Matrix{Int32}
		hashes = hashfn(randn(5, 10))
		@test isa(hashes, Matrix{Int32})
	end
end
