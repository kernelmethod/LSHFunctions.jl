using Test, Random, LSH

include(joinpath("..", "utils.jl"))

#==================
Tests
==================#
@testset "MIPSHash tests" begin
    Random.seed!(RANDOM_SEED)
    import LinearAlgebra: norm
    import Base.Iterators: product
    import SparseArrays: sprandn

    @testset "Can construct a simple MIPSHash" begin
        hashfn = MIPSHash()

        @test n_hashes(hashfn) == 1
        @test hashtype(hashfn) == Vector{Int32}
        @test isa(hashfn, MIPSHash{Float32})    # Default dtype should be Float32
        @test isa(hashfn, LSH.AsymmetricLSHFunction)

        ##
        hashfn = MIPSHash(12)

        @test n_hashes(hashfn) == 12

        ##
        hashfn = MIPSHash(; dtype=Float64)

        @test isa(hashfn, MIPSHash{Float64})

        ##
        hashfn = MIPSHash{Float64}()
        @test isa(hashfn, MIPSHash{Float64})

        ### Invalid hash function construction

        @test_throws ErrorException MIPSHash(-1)
        @test_throws ErrorException MIPSHash(; m=-1)
        @test_throws ErrorException MIPSHash(; m=0)
        @test_throws ErrorException MIPSHash(; scale=-1)
        @test_throws ErrorException MIPSHash(; scale=0)
    end

    @testset "Hashing returns the correct data types" begin
        hashfn = MIPSHash{Float64}(; scale=1, m=3)

        # Matrix{Float64} -> Matrix{Int32}
        x = randn(4, 10)
        p_hashes = index_hash(hashfn, x)
        q_hashes = query_hash(hashfn, x)

        @test isa(p_hashes, Matrix{Int32})
        @test isa(q_hashes, Matrix{Int32})
        @test Vector{eltype(p_hashes)} == hashtype(hashfn)
        @test Vector{eltype(q_hashes)} == hashtype(hashfn)

        # Vector{Float64} -> Vector{Int32}
        x = randn(4)
        p_hashes = index_hash(hashfn, x)
        q_hashes = query_hash(hashfn, x)

        @test isa(index_hash(hashfn, x), Vector{Int32})
        @test isa(query_hash(hashfn, x), Vector{Int32})
    end

    @testset "MIPSHash h(P(x)) is correctly computed" begin
        n_hashes = 128
        scale = 0.5
        m = 3
        hashfn = MIPSHash(n_hashes; scale=scale, m=m)

        @test size(hashfn.coeff_B) == (n_hashes, 3)
        @test size(hashfn.shift) == (n_hashes,)

        x = randn(20)
        hash = index_hash(hashfn, x)

        @test isa(hash, Vector{Int32})
        @test length(hash) == n_hashes

        # Since resize_pow2 was not specified, and x is the largest input seen so
        # far, the number of coefficients / hash should be equal to 20

        @test size(hashfn.coeff_A) == (n_hashes, length(x))

        ### Compute hash manually
        # Start by performing the transform P(x)
        coeff = [hashfn.coeff_A hashfn.coeff_B]
        u = x / norm(x)
        norm_powers = [norm(u)^2, norm(u)^4, norm(u)^8]
        Px = [u; norm_powers]

        # Now compute the L^2 hash of P(x)
        manual_hash = coeff * Px ./ scale .+ hashfn.shift
        manual_hash = floor.(Int32, manual_hash)

        @test manual_hash == hash
    end

    @testset "MIPSHash h(Q(x)) is correctly computed" begin
        n_hashes = 128
        scale = 0.5
        m = 3
        hashfn = MIPSHash(n_hashes; scale=scale, m=m)

        @test size(hashfn.coeff_B) == (n_hashes, m)
        @test size(hashfn.shift) == (n_hashes,)

        x = randn(40)
        hash = query_hash(hashfn, x)

        @test isa(hash, Vector{Int32})
        @test length(hash) == n_hashes

        # Since x is the largest input we've seen so far and resize_pow2 not
        # specified, the number of coefficients per hash in hashfn.coeff_A
        # should be equal to the length of x.

        @test size(hashfn.coeff_A) == (n_hashes, length(x))

        ### Compute hash manually
        u = x ./ norm(x)
        Qx = [u; 1/2; 1/2; 1/2]
        coeff = [hashfn.coeff_A hashfn.coeff_B]

        @test size(Qx) == (length(x)+3,)

        # Then, we compute the L^2 hash of Qx
        manual_hash = coeff * Qx ./ scale .+ hashfn.shift
        manual_hash = floor.(Int32, manual_hash)

        @test manual_hash == hash
    end

    @testset "Hash inputs of different sizes" begin
        n_hashes = 16
        hashfn = MIPSHash(n_hashes)

        index_hash(hashfn, rand(10))
        @test size(hashfn.coeff_A) == (n_hashes, 10)

        index_hash(hashfn, rand(14))
        @test size(hashfn.coeff_A) == (n_hashes, 14)

        index_hash(hashfn, rand(8))
        @test size(hashfn.coeff_A) == (n_hashes, 14)

        query_hash(hashfn, rand(10))
        @test size(hashfn.coeff_A) == (n_hashes, 14)

        query_hash(hashfn, rand(20))
        @test size(hashfn.coeff_A) == (n_hashes, 20)

        query_hash(hashfn, rand(100))
        @test size(hashfn.coeff_A) == (n_hashes, 100)
    end

    @testset "resize_pow2 increases number of coefficients to powers of 2" begin
        hashfn = MIPSHash(10; resize_pow2=true)
        @test size(hashfn.coeff_A) == (10, 0)

        index_hash(hashfn, rand(3))
        @test size(hashfn.coeff_A) == (10, 4)

        query_hash(hashfn, rand(2))
        @test size(hashfn.coeff_A) == (10, 4)

        query_hash(hashfn, rand(5))
        @test size(hashfn.coeff_A) == (10, 8)

        index_hash(hashfn, rand(7))
        @test size(hashfn.coeff_A) == (10, 8)
    end

    @testset "MIPSHash generates collisions for large inner products" begin
        n_hashes = 256
        scale = 1
        m = 5
        hashfn = MIPSHash(n_hashes; scale=scale, m=m)

        x = randn(20)
        x_query_hashes = query_hash(hashfn, x)

        # Check that MIPSHash isn't just generating a single query hash
        @test any(x_query_hashes .!= x_query_hashes[1])

        # Compute the indexing hashes for a dataset with four vectors:
        # a) 10 * x (where x is the test query vector)
        # b) x
        # c) A vector of all zeros
        # d) -x
        dataset = [(10*x) x zero(x) -x]
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
        hashfn = MIPSHash(8; scale=1, m=1)

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
