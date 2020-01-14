using Test, Random, LSH

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

        # Hash L^2([-1,1]) over L^p distance
        hashfn = ChebHash(ℓ_1)

        @test n_hashes(hashfn) == 1
        @test similarity(hashfn) == ℓ_1
        @test hashtype(hashfn) == hashtype(LSHFunction(ℓ_1))
    end

    #==========
    Cosine similarity hashing
    ==========#
    @testset "Hash cosine similarity with trivial inputs" begin
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

        @test embedded_similarity(hashfn, f, g) ≈ 0

        hf, hg = hashfn(f), hashfn(g)
        @test 0.45 ≤ mean(hf .== hg) ≤ 0.55

        ### Hash inputs with cosine similarity 1
        f(x) = sin(x)
        g(x) = 2f(x)

        @test embedded_similarity(hashfn, f, g) ≈ 1

        hf, hg = hashfn(f), hashfn(g)
        @test mean(hf .== hg) == 1
    end
end
