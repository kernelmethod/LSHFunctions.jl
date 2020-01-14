using Test, Random, LSH

include(joinpath("..", "utils.jl"))

#========================
Helper functions and types
========================#

# ShiftedSine and ShiftedCosine so that we can quickly construct functions of
# the form f(x) = cos(αx+δ) and g(x) = sin(αx+δ) without having to constantly
# generate new functions, which is fairly time-intensive.
struct ShiftedSine{S <: Real, T <: Real}
    α :: T
    δ :: T
end

struct ShiftedCosine{S <: Real, T <: Real}
    α :: T
    δ :: T
end

ShiftedSine(α::S, δ::T) where {S,T} = ShiftedSine{S,T}(α,δ)
ShiftedCosine(α::S, δ::T) where {S,T} = ShiftedCosine{S,T}(α,δ)

(f::ShiftedSine)(x)   = @. sin(f.α * x + f.δ)
(f::ShiftedCosine)(x) = @. cos(f.α * x + f.δ)

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
        hashfn = ChebHash(ℓ1)

        @test n_hashes(hashfn) == 1
        @test similarity(hashfn) == ℓ1
        @test hashtype(hashfn) == hashtype(LSHFunction(ℓ1))
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

    @testset "Hash cosine similarity with nontrivial inputs" begin
        # Construct pairs of trig functions f(x) = sin(πx+δx) and
        # g(y) = cos(πy+δy) and hash them on cosine similarity.
        # Note: use trig functions since they're fairly cheap to represent with
        # Chebyshev series.
        interval = LSH.@interval(-1.0 ≤ x ≤ 1.0)
        hashfn = ChebHash(cossim, 1024; interval=interval)

        trig_function_test() = begin
            f = ShiftedSine(π, randn())
            g = ShiftedCosine(π, randn())

            sim = cossim(f, g, interval)
            hx, hy = hashfn(f), hashfn(g)
            prob = LSH.single_hash_collision_probability(hashfn, sim)

            prob - 0.05 ≤ mean(hx .== hy) ≤ prob + 0.05
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
