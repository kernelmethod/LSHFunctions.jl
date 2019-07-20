using Test, Random, LSH

@testset "LSH tests" begin
	Random.seed!(0)

	@testset "MIPS hashing tests" begin
		import LSH: MIPSHash_P_LSH, MIPSHash_Q_LSH

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
			@test isa(MIPSHash_P_LSH(hashfn, x), Matrix{Int32})
			@test isa(MIPSHash_Q_LSH(hashfn, x), Matrix{Int32})

			# Vector{Float64} -> Matrix{Int32}
			x = randn(4)
			@test isa(MIPSHash_P_LSH(hashfn, x), Vector{Int32})
			@test isa(MIPSHash_Q_LSH(hashfn, x), Vector{Int32})
		end

		@test_skip @testset "Equivalent to L^2 hash when m == 0" begin
			input_length = 10
			n_hashes = 64
			denom = 2

			MIPS_hashfn = MIPSHash{Float32}(input_length, n_hashes, denom, 0)
			coeff_A, shift = MIPS_hashfn.coeff_A, MIPS_hashfn.shift
			L2_hashfn = LpDistHash{Float32,typeof(coeff_A)}(coeff_A, denom, shift)

			# When m == 0, we should have h(P(x)) == h(Q(x)) for the MIPS hash
			x = randn(input_length, 32)
			@test MIPSHash_P_LSH(MIPS_hashfn, x; scale=false) == MIPSHash_Q_LSH(MIPS_hashfn, x)

			# Moreover, we expect that h(P(x)) == h(Q(x)) == k(x), where k is the equivalent
			# L^2 hash function.
			@test MIPSHash_P_LSH(MIPS_hashfn, x; scale=false) == L2_hashfn(x)
		end
	end
end
