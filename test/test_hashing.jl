using Test, Random, LSH

@testset "LSH tests" begin
	Random.seed!(0)

	@testset "L^p hashing tests" begin
		@testset "Can construct a simple L^p distance hash function" begin
			input_length = 5
			n_hashes = 8
			denom = 2

			for N in (1,2)
				Lp_hash = LpDistHash{Float32,1}(input_length, n_hashes, denom)

				@test size(Lp_hash.coeff) == (n_hashes, input_length)
				@test Lp_hash.denom == denom
				@test size(Lp_hash.shift) == (n_hashes,)
			end
		end

		@testset "Type consistency in LpDistHash fields" begin
			# Should have type consistency between the fields of the struct,
			# so that we avoid expensive type conversions.
			Lp_hash = LpDistHash{Float32,1}(5, 5, 1)

			@test isa(Lp_hash.coeff, Array{Float32})
			@test isa(Lp_hash.denom, Float32)
			@test isa(Lp_hash.shift, Array{Float32})

			Lp_hash = LpDistHash{Float64,1}(5, 5, 1)
			@test isa(Lp_hash.coeff, Array{Float64})
			@test isa(Lp_hash.denom, Float64)
			@test isa(Lp_hash.shift, Array{Float64})
		end

		@testset "Hashes are correctly computed" begin
			input_length = 5
			n_hashes = 8
			denom = 2

			hashfn = LpDistHash{Float32,2}(input_length, n_hashes, denom)
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

	@testset "MIP hashing tests" begin
		@testset "Can construct a simple MIPS hash function" begin
			input_length = 5
			n_hashes = 8
			denom = 2
			m = 5

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

				@test isa(hashfn.coeff_A, Array{T})
				@test isa(hashfn.coeff_B, Array{T})
				@test isa(hashfn.shift, Array{T})
				@test isa(hashfn.Qshift, Array{T})
				@test isa(hashfn.denom, T)
			end
		end
	end
end
