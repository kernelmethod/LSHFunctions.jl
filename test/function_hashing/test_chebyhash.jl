using Test, Random, LSH

include(joinpath("..", "utils.jl"))

#========================
Tests
========================#

@testset "ChebHash tests" begin
    Random.seed!(RANDOM_SEED)

    @testset "Construct ChebHash" begin
        # Hash L^2([-1,1]) over cosine similarity.
        hashfn = ChebHash(CosSim, 5)

        @test similarity(hashfn) == CosSim
        @test n_hashes(hashfn) == 5
        @test hashtype(hashfn) == hashtype(LSHFunction(CosSim))

        # Hash L^2([-1,1]) over L^p distance
        hashfn = ChebHash(ℓ_1)

        @test n_hashes(hashfn) == 1
        @test similarity(hashfn) == ℓ_1
        @test hashtype(hashfn) == hashtype(LSHFunction(ℓ_1))
    end
end
