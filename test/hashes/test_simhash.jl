using Test, Random, LSHFunctions

include(joinpath("..", "utils.jl"))

#==================
Tests
==================#
@testset "SimHash tests" begin
    Random.seed!(RANDOM_SEED)

    @testset "Construct SimHash" begin
        # Create 128 SimHash hash functions
        hashfn = LSHFunctions.SimHash(128)

        @test n_hashes(hashfn) == 128
        @test hashtype(hashfn) == Bool
        @test similarity(hashfn) == cossim

        # By default, SimHash should start off unable to hash inputs of any
        # size, and should not resize in powers of 2.
        @test LSHFunctions.current_max_input_size(hashfn) == 0
        @test hashfn.resize_pow2 == false
    end

    @testset "Hash simple inputs" begin
        hashfn = SimHash(128)

        ## Test 1: a single input that is just the zero vector
        x = zeros(10)
        @test all(hashfn(x) .== true)

        ## Test 2: many inputs, all of which are zero vectors
        x = zeros(10, 32)
        @test all(hashfn(x) .== true)

        ## Test 3: many identical inputs
        u = randn(10)
        x = similar(u, 10, 32)
        x .= u
        hashes = hashfn(x)
        @test all(hashes[:,1] .== hashes)

        # For any input, the probability that all of its hashes are the
        # same should be extremely low, 2^(-n_hashes+1). Thus, there
        # should be at least one true bit and one false bit among
        # the hashes for an input.
        @test any(hashes[:,1]) && !all(hashes[:,1])
    end

    @testset "Hashing returns the correct data types" begin
        hashfn = SimHash(2)

        ## Test 1: Vector{Float64} -> BitArray{1}
        hashes = hashfn(randn(5))
        @test isa(hashes, BitArray{1})

        ## Test 2: Matrix{Float64} -> BitArray{2}
        hashes = hashfn(randn(5, 10))
        @test isa(hashes, BitArray{2})
    end

    @testset "SimHash collision probabilities match expectations" begin
        # Draw a massive number of hash functions, since we're going to be testing
        # the probability of collision for many different pairs of random inputs.
        # As the number of pairs increases, the probability that one pair has an
        # above- or below-average number of collisions increases.
        hashfn = LSHFunctions.SimHash(1024)

        # Run test_collision_probability lots of times to ensure that the
        # collision probability is close to what's advertised.
        # Dry run: just try once
        @test test_collision_probability(hashfn, 0.05)

        # Full test: test many more times
        @test all(test_collision_probability(hashfn, 0.05) for ii = 1:128)
    end

    @testset "SimHash current_max_input_size scales with input size" begin
        inputs = [rand(4), rand(7), rand(127), rand(4)]

        ### First round of tests: run with resize_pow2 == false
        hashfn = LSHFunctions.SimHash(1; resize_pow2 = false)

        for (ii,x) in enumerate(inputs)
            max_size_seen = inputs[1:ii] .|> length |> maximum
            hashfn(x)
            @test LSHFunctions.current_max_input_size(hashfn) == max_size_seen
        end

        # Second round of tests: run with resize_pow2 == true
        hashfn = LSHFunctions.SimHash(1; resize_pow2 = true)
        for (ii,x) in enumerate(inputs)
            max_size_seen = inputs[1:ii] .|> length |> maximum
            next_pow_2 = nextpow(2, max_size_seen)
            hashfn(x)
            @test LSHFunctions.current_max_input_size(hashfn) == next_pow_2
        end
    end
end
