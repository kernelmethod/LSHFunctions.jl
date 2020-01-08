using Test, Random, LSH
using LinearAlgebra: dot, norm

include("utils.jl")

#==================
Tests
==================#

@testset "Test similarity function API" begin
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

        (::mytype)(x,y) = CosSim(x,y)

        LSH.@register_similarity!(mytype(), SimHash)
        hashfn = LSH.LSHFunction(mytype())
        @test isa(hashfn, SimHash)
    end
end
