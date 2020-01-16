using Test, Random, LSH, LinearAlgebra, SparseArrays

include(joinpath("..", "utils.jl"))

#==================
Tests
==================#
@testset "SignALSH tests" begin
    Random.seed!(RANDOM_SEED)

    @testset "Construct SignALSH" begin
        hashfn = SignALSH(; maxnorm=1)
        @test n_hashes(hashfn) == 1
        @test isa(hashfn, SignALSH{Float32})
        @test isa(hashfn, LSH.AsymmetricLSHFunction)
        @test hashtype(hashfn) == Bool
        @test similarity(hashfn) == inner_prod

        hashfn = SignALSH(32; maxnorm=1)
        @test n_hashes(hashfn) == 32

        hashfn = SignALSH(; dtype=Float64, maxnorm=1)
        @test isa(hashfn, SignALSH{Float64})

        # n_hashes must be positive
        @test_throws ErrorException SignALSH(-1; maxnorm=1)
        @test_throws ErrorException SignALSH( 0; maxnorm=1)

        # maxnorm must be specified and non-negative
        @test_throws ErrorException SignALSH()
        @test_throws ErrorException SignALSH(; maxnorm=-1)

        # m must be positive
        @test_throws ErrorException SignALSH(; m=-1)
        @test_throws ErrorException SignALSH(; m=0)
    end

    @test_skip @testset "Can hash inputs correctly with SignALSH" begin
        input_length = 5
        n_hashes = 8

        hashfn = SignALSH(n_hashes; m=3)
        simhash = SimHash([hashfn.coeff_A hashfn.coeff_B])

        # By default, hashfn should be using Float32 for its coefficients
        @test isa(hashfn, SignALSH{Float32})

        X = rand(Float32, input_length, 128)
        ihashes = index_hash(hashfn, X)
        qhashes = query_hash(hashfn, X)

        @test eltype(ihashes) == eltype(qhashes)
        @test hashtype(hashfn) == Bool

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

    @testset "SignALSH can't hash inputs of norm > maxnorm" begin
        hashfn = SignALSH(; maxnorm=0)
        @test_throws ErrorException index_hash(hashfn, rand(4))
        @test_throws ErrorException query_hash(hashfn, rand(4))

        # Should have no issue if norm(x) == maxnorm
        @test index_hash(hashfn, zeros(4)) |> length == 1
        @test query_hash(hashfn, zeros(4)) |> length == 1
    end

    @testset "SignALSH generates collisions for large inner products" begin
        input_length = 5; n_hashes = 128;

        # Compare a random vector x against four other vectors:
        # a) 10 * x
        # b) x
        # c) A vector of all zeros
        # d) -x
        x = randn(input_length)
        x2, x3, x4 = 10*x, zero(x), -x

        maxnorm = (x, x2, x3, x4) .|> norm |> maximum
        hashfn = SignALSH(n_hashes; maxnorm=maxnorm)

        x_query_hashes = query_hash(hashfn, x)

        dataset = [x2 x x3 x4]
        p_hashes = index_hash(hashfn, dataset)

        # Each collection of hashes should be different from one another
        @test let result = true
            for (ii,jj) in Iterators.product(1:4, 1:4)
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
        @test n_collisions[1] > n_collisions[2] >
              n_collisions[3] > n_collisions[4]
    end

    @testset "Can hash sparse arrays" begin
        input_size = 100
        n_inputs = 150
        n_hashes = 2

        hashfn = SignALSH(n_hashes; maxnorm=4*input_size)
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
        hashfn = SignALSH(n_hashes; maxnorm=4*input_size)

        ## Test 1: regular matrix adjoint
        x = randn(n_inputs, input_size)'
        @test index_hash(hashfn, x) == index_hash(hashfn, copy(x))
        @test query_hash(hashfn, x) == query_hash(hashfn, copy(x))

        ## Test 2: sparse matrix adjoint
        x = sprandn(n_inputs, input_size, 0.2)'
        @test index_hash(hashfn, x) == index_hash(hashfn, copy(x))
        @test query_hash(hashfn, x) == query_hash(hashfn, copy(x))
    end

    @testset "Hash inputs of different sizes" begin
        n_hashes = 42
        hashfn = SignALSH(n_hashes; maxnorm=100)

        @test size(hashfn.coeff_A) == (n_hashes, 0)

        index_hash(hashfn, rand(10))
        @test size(hashfn.coeff_A) == (n_hashes, 10)

        query_hash(hashfn, rand(20))
        @test size(hashfn.coeff_A) == (n_hashes, 20)

        index_hash(hashfn, rand(5))
        @test size(hashfn.coeff_A) == (n_hashes, 20)

        query_hash(hashfn, rand(15))
        @test size(hashfn.coeff_A) == (n_hashes, 20)

        index_hash(hashfn, rand(20))
        @test size(hashfn.coeff_A) == (n_hashes, 20)

        query_hash(hashfn, rand(20))
        @test size(hashfn.coeff_A) == (n_hashes, 20)
    end

    @testset "Hash inputs of different sizes with resize_pow2 = true" begin
        n_hashes = 25
        hashfn = SignALSH(n_hashes; maxnorm=100, resize_pow2=true)

        @test size(hashfn.coeff_A) == (n_hashes, 0)

        index_hash(hashfn, rand(4))
        @test size(hashfn.coeff_A) == (n_hashes, 4)

        query_hash(hashfn, rand(8))
        @test size(hashfn.coeff_A) == (n_hashes, 8)

        index_hash(hashfn, rand(9))
        @test size(hashfn.coeff_A) == (n_hashes, 16)

        query_hash(hashfn, rand(17))
        @test size(hashfn.coeff_A) == (n_hashes, 32)

        index_hash(hashfn, rand(15))
        @test size(hashfn.coeff_A) == (n_hashes, 32)

        query_hash(hashfn, rand(31))
        @test size(hashfn.coeff_A) == (n_hashes, 32)

        index_hash(hashfn, rand(32))
        @test size(hashfn.coeff_A) == (n_hashes, 32)

        query_hash(hashfn, rand(32))
        @test size(hashfn.coeff_A) == (n_hashes, 32)
    end
end
