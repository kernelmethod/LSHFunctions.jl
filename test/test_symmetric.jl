using Test, Random, LSH

@testset "Test symmetric hash families" begin
	Random.seed!(0)

	@testset "L^p hashing tests" begin
		@testset "Can construct a simple L^p distance hash function" begin
			input_length = 5
			n_hashes = 8
			denom = 2

			for p in (1,2)
				Lp_hash = LpDistHash(input_length, n_hashes, denom, p)

				@test size(Lp_hash.coeff) == (n_hashes, input_length)
				@test Lp_hash.denom == denom
				@test size(Lp_hash.shift) == (n_hashes,)
			end
		end

		@testset "Type consistency in LpDistHash fields" begin
			# Should have type consistency between the fields of the struct,
			# so that we avoid expensive type conversions.
			Lp_hash = LpDistHash{Float32}(5, 5, 1)

			@test isa(Lp_hash.coeff, Array{Float32})
			@test isa(Lp_hash.denom, Float32)
			@test isa(Lp_hash.shift, Array{Float32})

			Lp_hash = LpDistHash{Float64}(5, 5, 1)
			@test isa(Lp_hash.coeff, Array{Float64})
			@test isa(Lp_hash.denom, Float64)
			@test isa(Lp_hash.shift, Array{Float64})
		end

		@testset "Hashes are correctly computed" begin
			input_length = 5
			n_hashes = 8
			denom = 2

			hashfn = LpDistHash{Float32}(input_length, n_hashes, denom)
			coeff, shift = hashfn.coeff, hashfn.shift

			# Test on a single input
			x = randn(input_length)
			hashes = hashfn(x)
			manual_hashes = floor.(Int32, (coeff * x .+ shift) ./ denom)

			@test isa(hashes, Vector{Int32})
			@test hashes == manual_hashes

			# Test on many inputs, simultaneously
			x = randn(input_length, 128)
			hashes = hashfn(x)
			manual_hashes = floor.(Int32, (coeff * x .+ shift) ./ denom)

			@test isa(hashes, Matrix{Int32})
			@test hashes == manual_hashes
		end
	end


end
