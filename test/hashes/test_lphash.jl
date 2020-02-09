using Test, Random, LSHFunctions

include(joinpath("..", "utils.jl"))

#==================
Tests
==================#
@testset "LpHash tests" begin
    Random.seed!(RANDOM_SEED)

    @testset "Can construct an ℓ^p distance hash function" begin
        # Construct a hash for L^1 distance
        L1_hash = L1Hash(5; scale = 2)
        @test n_hashes(L1_hash) == 5
        @test L1_hash.scale == 2
        @test L1_hash.power == 1
        @test similarity(L1_hash) == ℓ1
        @test hashtype(L1_hash) == Int32

        # Construct a hash for L^2 distance
        L2_hash = L2Hash(12; scale = 3.4)
        @test n_hashes(L2_hash) == 12
        @test L2_hash.scale == Float32(3.4)
        @test L2_hash.power == 2
        @test similarity(L2_hash) == ℓ2

        # Construct a hash using a specified dtype
        @test isa(L1Hash(1; dtype=Float32), LSHFunctions.LpHash{Float32})
        @test isa(L2Hash(1; dtype=Float64), LSHFunctions.LpHash{Float64})
    end

    @testset "Hashes are correctly computed" begin
        n_hashes = 8
        scale = 2

        hashfn = L2Hash(n_hashes; scale=scale)

        # Test on a single input
        x = randn(8)
        hashes = hashfn(x)
        manual_hashes = floor.(Int32, hashfn.coeff * x ./ scale .+ hashfn.shift)

        @test isa(hashes, Vector{Int32})
        @test hashes == manual_hashes

        # Test on many inputs, simultaneously
        x = randn(8, 128)
        hashes = hashfn(x)
        manual_hashes = floor.(Int32, hashfn.coeff * x ./ scale .+ hashfn.shift)

        @test isa(hashes, Matrix{Int32})
        @test hashes == manual_hashes
    end

    @testset "Hashes have the correct dtype" begin
        hashfn = L1Hash(5)

        # Test 1: Vector{Float64} -> Vector{Int32}
        hashes = hashfn(randn(5))
        @test eltype(hashes) == hashtype(hashfn)
        @test isa(hashes, Vector{Int32})

        # Test 2: Matrix{Float64} -> Matrix{Int32}
        hashes = hashfn(randn(5, 10))
        @test isa(hashes, Matrix{Int32})
    end

    @testset "Nearby points experience more frequent collisions" begin
        hashfn = L2Hash(1024; dtype=Float64, scale=4)

        x1 = randn(128)
        x2 = x1 + 0.05 * randn(length(x1))
        x3 = x1 + 0.50 * randn(length(x1))
        x4 = x1 + 2.00 * randn(length(x1))

        h1, h2, h3, h4 = hashfn(x1), hashfn(x2), hashfn(x3), hashfn(x4)
        @test sum(h1 .== h4) < sum(h1 .== h3) < sum(h1 .== h2)
    end

    @testset "Hash collision frequency matches probability" begin
        hashfn = L2Hash(1024; scale = 4)

        # Dry run
        @test test_collision_probability(hashfn, 0.05)

        # Full test
        @test all(test_collision_probability(hashfn, 0.05) for ii = 1:128)
    end

    @testset "Hash inputs of different sizes" begin
        n_hashes = 10
        for hashfn_type in (:L1Hash, :L2Hash)
            hashfn = eval(hashfn_type)(n_hashes)
            @test size(hashfn.coeff) == (n_hashes, 0)

            hashfn(rand(5))
            @test size(hashfn.coeff) == (n_hashes, 5)

            hashfn(rand(20))
            @test size(hashfn.coeff) == (n_hashes, 20)

            hashfn(rand(10))
            @test size(hashfn.coeff) == (n_hashes, 20)
        end
    end

    @testset "Hash inputs of different sizes with resize_pow2 = true" begin
        n_hashes = 20
        for hashfn_type in (:L1Hash, :L2Hash)
            hashfn = eval(hashfn_type)(n_hashes; resize_pow2=true)
            @test size(hashfn.coeff) == (n_hashes, 0)

            hashfn(rand(4))
            @test size(hashfn.coeff) == (n_hashes, 4)

            hashfn(rand(5))
            @test size(hashfn.coeff) == (n_hashes, 8)

            hashfn(rand(10))
            @test size(hashfn.coeff) == (n_hashes, 16)

            hashfn(rand(8))
            @test size(hashfn.coeff) == (n_hashes, 16)
        end
    end

    @testset "collision_probability works correctly" begin
        hashfn = L1Hash()

        # collision_probability should be 1 for two inputs of distance zero
        x = rand(4)
        @test collision_probability(hashfn, x, x) ≈ 1.0

        # collision_probability with n_hashes=N should be the same as
        # collision_probability with n_hashes=1, raised to the power N
        y = rand(4)
        @test collision_probability(hashfn, x, y; n_hashes=10) ≈
              collision_probability(hashfn, x, y; n_hashes=1)^10
    end
end


