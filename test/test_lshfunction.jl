#==================

Tests for the LSHFunction() function for constructing hash functions.

==================#

using Test, Random, LSH

include("utils.jl")

#==================
Tests
==================#

@testset "Test LSHFunction()" begin
    Random.seed!(RANDOM_SEED)

    @testset "Create cosine similarity hash function" begin
        hashfn = LSHFunction(CosSim)

        @test similarity(hashfn) == CosSim
        @test n_hashes(hashfn) == 1
        @test isa(hashfn, SimHash)
    end

    @testset "Create L^p distance hash function" begin
        hashfn = LSHFunction(ℓ_1, 20; r = 4.0)

        @test similarity(hashfn) == ℓ_1
        @test n_hashes(hashfn) == 20
        @test isa(hashfn, LSH.LpHash)
        @test hashfn.r == 4.0
    end

    @testset "Create Jaccard similarity hash function" begin
        hashfn = LSHFunction(Jaccard, 64)

        @test similarity(hashfn) == Jaccard
        @test n_hashes(hashfn) == 64
        @test isa(hashfn, MinHash)
    end

    @testset "Call LSHFunction() with invalid similarity function" begin
        import LinearAlgebra: dot, norm
        my_CosSim(x,y) = dot(x,y) / (norm(x) * norm(y))

        @test_throws(MethodError, LSHFunction(my_CosSim))
    end
end
