using Test, Random, LSHFunctions
using LinearAlgebra: norm
using QuadGK

include(joinpath("..", "utils.jl"))

#========================
Tests
========================#

@testset "ChebHash tests" begin
    Random.seed!(RANDOM_SEED)

    @testset "Construct ChebHash" begin
        # Hash L^2([-1,1]) over cosine similarity.
        hashfn = ChebHash(cossim, 5)

        @test similarity(hashfn) == cossim
        @test n_hashes(hashfn) == 5
        @test hashtype(hashfn) == hashtype(LSHFunction(cossim))

        # Hash L^2([-1,1]) over L^2 distance
        hashfn = ChebHash(L2)

        @test n_hashes(hashfn) == 1
        @test similarity(hashfn) == L2
        @test hashtype(hashfn) == hashtype(LSHFunction(ℓ2))
    end

    @testset "Provide invalid similarity" begin
        # When we pass in a similarity that is not supported by ChebHash
        # we should receive an error.
        @test_throws(ErrorException, ChebHash((x,y) -> abs(x-y)))
        @test_throws(ErrorException, ChebHash(ℓ1))
        @test_throws(ErrorException, ChebHash(L1))
        @test_throws(ErrorException, ChebHash(ℓ2))

        # Construct a hash function (with valid similarity) in the same
        # manner as we did above in case the ChebHash API ever changes.
        # This ensures that we won't forget to update these tests.
        _ = ChebHash(L2)
    end

    #==========
    Cosine similarity hashing
    ==========#
    @testset "Hash cosine similarity (trivial inputs)" begin
        ### Hash inputs with cosine similarity -1
        f(x) = sin(x)
        g(x) = -f(x)
        hashfn = ChebHash(cossim, 1024)

        @test embedded_similarity(hashfn, f, g) ≈ -1

        hf, hg = hashfn(f), hashfn(g)
        @test mean(hf .== hg) == 0

        ### Hash inputs with cosine similarity 0
        # Note: f(x)g(x) = sin(x)cos(x) = 0.5 sin(2x), so that cosine
        # similarity is zero (over the interval [-1,1]).
        f(x) = sin(x)
        g(x) = cos(x)

        @test isapprox(embedded_similarity(hashfn, f, g), 0; atol=1e-15)

        hf, hg = hashfn(f), hashfn(g)
        @test 0.45 ≤ mean(hf .== hg) ≤ 0.55

        ### Hash inputs with cosine similarity 1
        f(x) = sin(x)
        g(x) = 2f(x)

        @test embedded_similarity(hashfn, f, g) ≈ 1

        hf, hg = hashfn(f), hashfn(g)
        @test mean(hf .== hg) == 1
    end

    @testset "Hash cosine similarity (nontrivial inputs)" begin
        interval = LSHFunctions.@interval(-1.0 ≤ x ≤ 1.0)
        hashfn = ChebHash(cossim, 1024; interval=interval)

        trig_function_test() = begin
            f = ShiftedSine(π, π * rand())
            g = ShiftedSine(π, π * rand())

            sim = cossim(f, g, interval)
            hf, hg = hashfn(f), hashfn(g)
            prob = collision_probability(hashfn, sim; n_hashes=1)

            prob - 0.05 ≤ mean(hf .== hg) ≤ prob + 0.05
        end

        # Dry-run: test on a single pair of inputs
        @test trig_function_test()

        # Full test: run across many pairs of inputs
        @test let success = true, ii = 1
            while ii ≤ 128 && success
                success = success && trig_function_test()
                ii += 1
            end
            success
        end
    end

    #==========
    L^2 distance hashing
    ==========#
    @testset "Hash L^2 distance (trivial inputs)" begin
        ### Hash two functions with L^2 distance ≈ 0
        f(x) = 0.0
        g(x) = (-0.5 ≤ x ≤ 0.5) ? 1e-3 : 0.0
        hashfn = ChebHash(L2, 1024)

        @test embedded_similarity(hashfn, f, g) ≈ 1e-3

        hf, hg = hashfn(f), hashfn(g)
        @test mean(hf .== hg) ≥ 0.95

        ### Hash two functions with large L^2 distance
        g(x) = (-0.5 ≤ x ≤ 0.5) ? 1e3 : 0.0

        @test embedded_similarity(hashfn, f, g) ≈ 1e3

        hf, hg = hashfn(f), hashfn(g)
        @test mean(hf .== hg) ≤ 0.02
    end

    @testset "Hash L^2 distance (nontrivial inputs)" begin
        interval = LSHFunctions.@interval(-1.0 ≤ x ≤ 1.0)
        hashfn = ChebHash(L2, 1024; interval=interval)

        trig_function_test() = begin
            f = ShiftedSine(π, π * rand())
            g = ShiftedSine(π, π * rand())

            sim = L2(f, g, interval)
            hf, hg = hashfn(f), hashfn(g)
            prob = collision_probability(hashfn, sim; n_hashes=1)

            prob - 0.05 ≤ mean(hf .== hg) ≤ prob + 0.05
        end

        # Dry-run: test on a single pair of inputs
        @test trig_function_test()

        # Full test: run across many pairs of inputs
        @test let success = true, ii = 1
            while ii ≤ 128 && success
                success = success && trig_function_test()
                ii += 1
            end
            success
        end
    end
end
