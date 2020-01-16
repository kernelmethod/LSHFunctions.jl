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
        hashfn = MonteCarloHash(cossim, μ)

        @test n_hashes(hashfn) == 1
        @test hashfn.μ == μ
        @test similarity(hashfn) == cossim
        @test hashtype(hashfn) == hashtype(LSHFunction(cossim))

        # Hash L^1([0,1]) over L^1 distance
        hashfn = MonteCarloHash(ℓ1, μ, 10)

        @test n_hashes(hashfn) == 10
        @test similarity(hashfn) == ℓ1
        @test hashtype(hashfn) == hashtype(LSHFunction(ℓ1))
    end

    #==========
    Cosine similarity hashing
    ==========#
    @testset "Hash cosine similarity with trivial inputs" begin
        # Hash over cosine similarity between two functions with cosine similarity
        # zero. The collision rate should be close to 50%.
        f(x) = (0.0 ≤ x ≤ 0.5) ? 1.0 : 0.0;
        g(x) = (0.0 ≤ x ≤ 0.5) ? 0.0 : 1.0;
        hashfn = MonteCarloHash(cossim, rand, 1024)

        @test embedded_similarity(hashfn, f, g) == 0.0

        hf, hg = hashfn(f), hashfn(g)
        @test 0.45 ≤ mean(hf .== hg) ≤ 0.55
    end

    @testset "Hash cosine similarity with nontrivial inputs" begin
        # Test hashing on cosine similarity using step functions defined
        # over [0,N]. The functions are piecewise constant on the intervals
        # [i,i+1], i = 1, ..., N.
        N = 10
        μ() = N * rand()
        hashfn = MonteCarloHash(cossim, μ, 2048)

        @test let success = true, ii = 1
            while success && ii ≤ 128
                f, f_steps = create_step_function(N)
                g, g_steps = create_step_function(N)

                prob = LSH.single_hash_collision_probability(hashfn, cossim(f_steps, g_steps))
                hf, hg = hashfn(f), hashfn(g)

                success &= (prob-0.05 ≤ mean(hf .== hg) ≤ prob+0.05)
                ii += 1
            end
            success
        end
    end

    #==========
    L^p distance hashing
    ==========#
    @testset "Hash L^p distance" begin
        # Use step functions to test distances between functions again, since
        # you can map them into R^N isomorphically
        N = 4
        μ() = N * rand()
        hashfn = MonteCarloHash(ℓ1, μ, 1024; volume = N)

        @test let success = true, ii = 1
            while success && ii ≤ 128
                f, f_steps = create_step_function(N)
                g, g_steps = create_step_function(N)

                # Hash collision rate should be close to the probability of
                # collision
                prob = LSH.single_hash_collision_probability(hashfn,
                                                             ℓ1(f_steps, g_steps))
                hf, hg = hashfn(f), hashfn(g)

                success &= (prob-0.05 ≤ mean(hf .== hg) ≤ prob+0.05)
                ii += 1
            end
            success
        end
    end
end
