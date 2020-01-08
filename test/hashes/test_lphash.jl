using Test, Random, LSH

include(joinpath("..", "utils.jl"))

#==================
Tests
==================#
@testset "LpHash tests" begin
    Random.seed!(RANDOM_SEED)
    import LSH: SymmetricLSHFunction

    @testset "Can construct an ℓ^p distance hash function" begin
        # Construct a hash for L^1 distance
        L1_hash = L1Hash(5; r = 2)
        @test n_hashes(L1_hash) == 5
        @test L1_hash.r == 2
        @test L1_hash.power == 1
        @test similarity(L1_hash) == ℓ_1

        # Construct a hash for L^2 distance
        L2_hash = L2Hash(12; r = 3.4)
        @test n_hashes(L2_hash) == 12
        @test L2_hash.r == Float32(3.4)
        @test L2_hash.power == 2
        @test similarity(L2_hash) == ℓ_2

        # Construct a hash using a specified dtype
        @test isa(L1Hash(1; dtype=Float32), LSH.LpHash{Float32})
        @test isa(L2Hash(1; dtype=Float64), LSH.LpHash{Float64})
    end

    @testset "Hashes are correctly computed" begin
        n_hashes = 8
        r = 2

        hashfn = L2Hash(n_hashes; r = r)

        # Test on a single input
        x = randn(8)
        hashes = hashfn(x)
        manual_hashes = floor.(Int32, hashfn.coeff * x ./ r .+ hashfn.shift)

        @test isa(hashes, Vector{Int32})
        @test hashes == manual_hashes

        # Test on many inputs, simultaneously
        x = randn(8, 128)
        hashes = hashfn(x)
        manual_hashes = floor.(Int32, hashfn.coeff * x ./ r .+ hashfn.shift)

        @test isa(hashes, Matrix{Int32})
        @test hashes == manual_hashes
    end

    @testset "Hashes have the correct dtype" begin
        hashfn = L1Hash(5)

        # Test 1: Vector{Float64} -> Vector{Int32}
        hashes = hashfn(randn(5))
        @test Vector{eltype(hashes)} == hashtype(hashfn)
        @test isa(hashes, Vector{Int32})

        # Test 2: Matrix{Float64} -> Matrix{Int32}
        hashes = hashfn(randn(5, 10))
        @test isa(hashes, Matrix{Int32})
    end

    @testset "Nearby points experience more frequent collisions" begin
        hashfn = L2Hash(1024; dtype=Float64, r=4)

        x1 = randn(128)
        x2 = x1 + 0.05 * randn(length(x1))
        x3 = x1 + 0.50 * randn(length(x1))
        x4 = x1 + 2.00 * randn(length(x1))

        h1, h2, h3, h4 = hashfn(x1), hashfn(x2), hashfn(x3), hashfn(x4)
        @test sum(h1 .== h4) < sum(h1 .== h3) < sum(h1 .== h2)
    end

    @testset "Hash collision frequency matches probability" begin
        hashfn = L2Hash(1024, r = 4)

        # Dry run
        @test test_collision_probability(hashfn, 0.05)

        # Full test
        @test all(test_collision_probability(hashfn, 0.05) for ii = 1:128)
    end
end
