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

        # Hash L^1([0,1]) over L^1 distance
        hashfn = MonteCarloHash(ℓ_1, μ, 10)

        @test n_hashes(hashfn) == 10
        @test similarity(hashfn) == ℓ_1
        @test hashtype(hashfn) == hashtype(LSHFunction(ℓ_1))
    end

    @testset "Hash cosine similarity with trivial inputs" begin
        # Hash over cosine similarity between two functions with cosine similarity
        # zero. The collision rate should be close to 50%.
        f(x) = (0.0 ≤ x ≤ 0.5) ? 1.0 : 0.0;
        g(x) = (0.0 ≤ x ≤ 0.5) ? 0.0 : 1.0;
        hashfn = MonteCarloHash(CosSim, rand, 1024)

        @test embedded_similarity(hashfn, f, g) == 0.0

        hf, hg = hashfn(f), hashfn(g)
        @test 0.45 ≤ mean(hf .== hg) ≤ 0.55
    end

    @testset "Hash cosine similarity with nontrivial inputs" begin
        # Test hashing on cosine similarity using step functions defined
        # over [0,N]. The functions are piecewise constant on the intervals
        # [i,i+1], i = 1, ..., N.
        N = 10
        f, f_steps = create_step_function(N)
        g, g_steps = create_step_function(N)

        μ() = N * rand()
        hashfn = MonteCarloHash(CosSim, μ, 1024)

        # The "embedded similarity" (effectively a Monte Carlo estimate of the
        # true similarity) should be close to the true cosine similarity between
        # f and g.
        true_sim = CosSim(f_steps, g_steps)
        embedded = embedded_similarity(hashfn, f, g)
        @test true_sim-0.05 ≤ embedded ≤ true_sim+0.05

        # Hash collision rate should be close to the probability of collision
        prob = LSH.single_hash_collision_probability(hashfn, true_sim)
        hf, hg = hashfn(f), hashfn(g)

        @test prob-0.05 ≤ mean(hf .== hg) ≤ prob+0.05
    end
end



