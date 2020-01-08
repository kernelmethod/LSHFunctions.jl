using Test, Random, LSH

include(joinpath("..", "utils.jl"))

#========================
Tests
========================#

@testset "Monte Carlo function hashing tests" begin
    Random.seed!(RANDOM_SEED)

    @testset "Construct MonteCarloHash" begin
        # Hash L^2([0,1]) over cosine similarity
        # Sampler μ() chooses a point in [0,1] uniformly at random
        μ() = rand()
        hashfn = MonteCarloHash(CosSim, μ)

        @test n_hashes(hashfn) == 1
        @test hashfn.μ == μ
        @test similarity(hashfn) == CosSim
        @test hashtype(hashfn) == hashtype(LSHFunction(CosSim))
    end
end
