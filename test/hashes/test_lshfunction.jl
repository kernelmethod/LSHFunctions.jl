#==================

Tests for the LSHFunction() function for constructing hash functions.

==================#

using Test, Random, LSH

include(joinpath("..", "utils.jl"))

#==================
Tests
==================#

@testset "Test similarity function registration" begin
    Random.seed!(RANDOM_SEED)

    @testset "Register a custom similarity" begin
        ### Test 1: register a function with @register_similarity!
        mysim(x,y) = dot(x,y) / (norm(x) * norm(y))

        @test_throws MethodError LSHFunction(mysim)
        @test_throws MethodError lsh_family(mysim)

        LSH.@register_similarity!(mysim, SimHash)

        hashfn = LSHFunction(mysim)
        @test isa(hashfn, SimHash)
        @test lsh_family(mysim) == SimHash

        ### Test 2: register a function-like object with @register_similarity!
        struct mytype end

        @test_throws MethodError LSHFunction(mytype())
        @test_throws MethodError lsh_family(mytype())
        @test_throws ErrorException LSH.@register_similarity!(mytype(), SimHash)

        (::mytype)(x,y) = cossim(x,y)

        LSH.@register_similarity!(mytype(), SimHash)
        hashfn = LSH.LSHFunction(mytype())
        @test isa(hashfn, SimHash)
        @test lsh_family(mysim) == SimHash

        ### Test 3: reset the available similarity functions
        LSH.@reset_similarities!()

        @test_throws MethodError LSHFunction(mysim)
        @test_throws MethodError LSHFunction(mytype())
    end
end

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
