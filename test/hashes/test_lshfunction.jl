#==================

Tests for the LSHFunction() function for constructing hash functions.

==================#

using Test, Random, LSH

include(joinpath("..", "utils.jl"))

#==================
Tests
==================#

@testset "Test LSHFunction()" begin
    Random.seed!(RANDOM_SEED)

    @testset "Create cosine similarity hash function" begin
        hashfn = LSHFunction(cossim)

        @test similarity(hashfn) == cossim
        @test n_hashes(hashfn) == 1
        @test isa(hashfn, SimHash)
    end

    @testset "Create L^p distance hash function" begin
        hashfn = LSHFunction(ℓ1, 20; r = 4.0)

        @test similarity(hashfn) == ℓ1
        @test n_hashes(hashfn) == 20
        @test isa(hashfn, LSH.LpHash)
        @test hashfn.r == 4.0
    end

    @testset "Create Jaccard similarity hash function" begin
        hashfn = LSHFunction(jaccard, 64)

        @test similarity(hashfn) == jaccard
        @test n_hashes(hashfn) == 64
        @test isa(hashfn, MinHash)
    end

    @testset "Create inner product similarity hash function" begin
        hashfn = LSHFunction(inner_prod, 10; maxnorm=1)

        @test similarity(hashfn) == inner_prod
        @test n_hashes(hashfn) == 10
        @test isa(hashfn, SignALSH)

        # Test that same exceptions are thrown for invalid construction of
        # SignALSH
        @test_throws ErrorException LSHFunction(inner_prod, -1; maxnorm=10)
        @test_throws ErrorException LSHFunction(inner_prod; m=0, maxnorm=10)
        @test_throws ErrorException LSHFunction(inner_prod)
    end

    @testset "Call LSHFunction() with invalid similarity function" begin
        import LinearAlgebra: dot, norm
        my_cossim(x,y) = dot(x,y) / (norm(x) * norm(y))

        @test_throws(MethodError, LSHFunction(my_cossim))
    end
end
