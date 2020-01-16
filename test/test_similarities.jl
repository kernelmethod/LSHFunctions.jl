using Test, Random, LSH, QuadGK
using LinearAlgebra: dot, norm

include("utils.jl")

#==================
Tests
==================#

@test_skip @testset "Test similarity function API" begin
    Random.seed!(RANDOM_SEED)

    @testset "Register a custom similarity" begin
        # Test 1: register a function with @register_similarity!
        mysim(x,y) = dot(x,y) / (norm(x) * norm(y))

        @test_throws(MethodError, LSH.LSHFunction(mysim))

        LSH.@register_similarity!(mysim, SimHash)

        hashfn = LSHFunction(mysim)
        @test isa(hashfn, SimHash)

        # Test 2: register a function-like object with @register_similarity!
        struct mytype end

        @test_throws(MethodError, LSH.LSHFunction(mytype()))
        @test_throws(ErrorException, LSH.@register_similarity!(mytype(), SimHash))

        (::mytype)(x,y) = cossim(x,y)

        LSH.@register_similarity!(mytype(), SimHash)
        hashfn = LSH.LSHFunction(mytype())
        @test isa(hashfn, SimHash)
    end
end

@testset "Function space L^p distance and norm" begin
    Random.seed!(RANDOM_SEED)

    @testset "Compute L^1 distance and norm" begin
        interval = LSH.@interval(-π ≤ x ≤ π)
        f(x) = 0
        g(x) = 2

        @test L1_norm(g, interval) ≈ L1(f, g, interval) ≈ 4π

        g(x) = x

        @test L1_norm(g, interval) ≈ L1(f, g, interval) ≈ π^2

        f(x) = x
        g(x) = 2x.^2

        @test L1(f, g, interval) ≈ L1_norm(x -> f(x) - g(x), interval)
        @test L1(f, g, interval) ≈ quadgk(x -> abs(f(x) - g(x)), -π, π)[1]
    end

    @testset "Compute L^2 distance and norm" begin
        interval = LSH.@interval(-π ≤ x ≤ π)
        f(x) = 0
        g(x) = 2

        @test L2_norm(g, interval) ≈ L2(f, g, interval) ≈ √(8π)

        g(x) = x

        @test L2_norm(g, interval) ≈ L2(f, g, interval) ≈ √(2π^3 / 3)

        f(x) = x
        g(x) = 2x.^2

        @test L2(f, g, interval) ≈ L2_norm(x -> f(x) - g(x), interval)
        @test L2(f, g, interval) ≈ √quadgk(x -> abs2(f(x) - g(x)), -π, π)[1]
    end

    @testset "Compute L^p distance and norm" begin
        interval = LSH.@interval(-π ≤ x ≤ π)
        f(x) = 0
        g(x) = 2

        @test Lp_norm(g, interval, 1) ≈ Lp(f, g, interval, 1) ≈ L1(f, g, interval)
        @test Lp_norm(g, interval, 2) ≈ Lp(f, g, interval, 2) ≈ L2(f, g, interval)
        @test Lp_norm(g, interval, 3) ≈ Lp(f, g, interval, 3) ≈ (16π)^(1/3)

        g(x) = x

        @test Lp_norm(g, interval, 1) ≈ Lp(f, g, interval, 1) ≈ L1(f, g, interval)
        @test Lp_norm(g, interval, 2) ≈ Lp(f, g, interval, 2) ≈ L2(f, g, interval)
        @test Lp_norm(g, interval, 3) ≈ Lp(f, g, interval, 3) ≈ (π^4/2)^(1/3)

        f(x) = x
        g(x) = 2x.^2
        p = rand() + 1

        @test Lp(f, g, interval, p) ≈ Lp_norm(x -> f(x) - g(x), interval, p)
        @test Lp(f, g, interval, p) ≈
              quadgk(x -> abs(f(x) - g(x))^p, -π, π)[1]^(1/p)
    end
end
