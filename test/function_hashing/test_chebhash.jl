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
        # Hash over cosine similarity with two functions with cosine similarity
        # zero. Their collision rate should be close to 50%.
        f(x) = (-1.0 ≤ x ≤ 0.0) ? 1.0 : 0.0;
        g(x) = (-1.0 ≤ x ≤ 0.0) ? 0.0 : 1.0;
        hashfn = ChebHash(cossim, 1024)

        @test embedded_similarity(hashfn, f, g) ≈ 0

        hf, hg = hashfn(f), hashfn(g)
        @test 0.45 ≤ mean(hf .== hg) ≤ 0.55
    end

    @testset "Hash cosine similarity with nontrivial inputs" begin
        # Test hashing on cosine similarity using step functions defined
        # over [0,N].
        N = 10
        interval = LSH.@interval(0.0 ≤ x ≤ N)
        hashfn = ChebHash(cossim, 1024; interval=interval)

        @test let success = true
            for ii = 1:128
                f, f_steps = create_step_function(N)
                g, g_steps = create_step_function(N)

                prob = LSH.single_hash_collision_probability(hashfn, cossim(f_steps, g_steps))
                hf, hg = hashfn(f), hashfn(g)

                success &= (prob-0.05 ≤ mean(hf .== hg) ≤ prob+0.05)

                if !success
                    break
                end
                success
            end
        end
    end
end



