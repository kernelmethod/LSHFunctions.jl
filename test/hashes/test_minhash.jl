using Test, Random, LSH

include(joinpath("..", "utils.jl"))

#==================
Tests
==================#
@testset "MinHash tests" begin
    Random.seed!(RANDOM_SEED)

    @testset "Can construct a MinHash hash function" begin
        symbol_collections = [
            collect(1:100),
            Set(1:30),
            ["a", "a", "b", "c"]
        ]
        n_hashes = [1, 8, 4]

        for (symbols, nh) in zip(symbol_collections, n_hashes)
            hashfn = MinHash(nh; symbols=symbols)

            @test isa(hashfn, MinHash{eltype(symbols)})
            @test LSH.n_hashes(hashfn) == nh
        end
    end

    @testset "MinHash range is 1:length(symbols)" begin
        symbols = collect(1:10)
        hashfn = MinHash(1000; symbols = symbols)

        # Verify that all hashes computed by hashfn are in the appropriate range
        has_correct_output_types = true
        has_correct_output_ranges = true

        for ii = 1:100
            if !(has_correct_output_types && has_correct_output_ranges)
                break
            end

            n = mod(rand(Int32), length(symbols)) + 1
            x = shuffle(symbols)[1:n]
            hashes = hashfn(x)

            has_correct_output_types  &= isa(hashes, Vector{UInt32})
            has_correct_output_ranges &= all(1 .≤ hashes .≤ 10 - (n - 1))
        end

        @test has_correct_output_types
        @test has_correct_output_ranges
    end
    
    @testset "Hashes are correctly computed" begin
        symbols = collect(50:100)
        hashfn = MinHash(10; symbols = symbols)

        dataset = shuffle(symbols)[1:10]
        hashes = hashfn(dataset)
        hashes_match = true

        for (hash, mapping) in zip(hashes, hashfn.mappings)
            if !hashes_match
                break
            end

            # Compute MinHash manually
            expected_hash = minimum(mapping[x] for x in dataset)
            hashes_match &= (hash == expected_hash)
        end

        @test hashes_match
    end

    @testset "Collision probabilities correlated with jaccard similarity" begin
        # Create three different datasets, such that the first and second datasets
        # have high jaccard similarity while the first and third have low jaccard
        # similarity. Test whether we in fact get higher collision rates for the
        # two sets with high similarity than the two sets with low similarity.
        symbols = collect(1:200)
        n_hashes = 100
        hashfn = MinHash(n_hashes; symbols=symbols)

        shuffle!(symbols)
        dataset_1 = symbols[1:100]
        dataset_2 = [symbols[1:75]; symbols[101:125]]
        dataset_3 = [symbols[1:25]; symbols[101:175]]

        hashes_1 = hashfn(dataset_1)
        hashes_2 = hashfn(dataset_2)
        hashes_3 = hashfn(dataset_3)

        @test sum(hashes_1 .== hashes_2) > sum(hashes_1 .== hashes_3)
    end

    @testset "Collision probability approx. equals jaccard similarity" begin
        # In theory, the probability of collision for two datasets should be
        # roughly equal to the jaccard similarity between those datasets
        symbols = collect(1:200)
        n_hashes = 10_000
        hashfn = MinHash(n_hashes; symbols=symbols)

        shuffle!(symbols)
        dataset_1 = Set(symbols[1:100])
        dataset_2 = Set([symbols[1:75]; symbols[101:125]])
        dataset_3 = Set([symbols[1:25]; symbols[101:175]])

        hashes_1 = hashfn(dataset_1)
        hashes_2 = hashfn(dataset_2)
        hashes_3 = hashfn(dataset_3)

        sim_12 = jaccard(Set(dataset_1), Set(dataset_2))
        sim_13 = jaccard(Set(dataset_1), Set(dataset_3))

        mean(x) = sum(x) / length(x)

        @test abs(mean(hashes_1 .== hashes_2) - sim_12) ≤ 0.01
        @test abs(mean(hashes_1 .== hashes_3) - sim_13) ≤ 0.01
    end

    @testset "Can omit symbol set to lazily update hash functions" begin
        # If the users want, they can decide not to pre-specify the symbol
        # set. In that case, the hash functions should be lazily updated.
        hashfn = MinHash(100)

        @test isa(hashfn, MinHash{Any, UInt64})

        symbols = [collect(1:10); "a"; "b"; "c"]
        dataset = shuffle(symbols)[1:3]
        hashes = hashfn(dataset)
        
        @test isa(hashes, Vector{UInt64})

        # Should still be able to hash new elements, updating the hash
        # functions further.
        new_dataset = ["d"; "e"; "f"]
        new_hashes = hashfn(new_dataset)

        @test sum(new_hashes .== hashes) == 0
    end

    @testset "Achieve collision probabilities with lazily updated hash functions" begin
        symbols = collect(1:1_000)
        dataset_1 = Set(shuffle(symbols)[1:500])
        dataset_2 = Set(shuffle(symbols)[1:500])
        similarity = jaccard(dataset_1, dataset_2)

        n_hashes = 10_000
        hashfn = MinHash(n_hashes; dtype=Int64)
        hashes_1 = hashfn(dataset_1)
        hashes_2 = hashfn(dataset_2)

        collision_proportion = sum(hashes_1 .== hashes_2) / n_hashes
        @test abs(collision_proportion - similarity) ≤ 0.01
    end
end
