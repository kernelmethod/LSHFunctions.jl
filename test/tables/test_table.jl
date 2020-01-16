#=
Tests for the LSHTable API
=#

using Test, Random, LSH

@testset "LSHTable tests" begin
    Random.seed!(0)

    @testset "Can construct an LSHTable over SimHash" begin
        n_hashes = 5

        hashfn = SimHash(n_hashes)
        table = LSHTable(hashfn)

        # Insert a random vector into the table
        x = rand(20)
        h = hashfn(x)
        insert!(table, x, "hello, world")

        @test haskey(table, h)
        @test table |> keys |> collect == [h]
        @test table[x] == ["hello, world"]

        x = 2 .* x
        insert!(table, x, 1234)

        @test table |> keys |> collect == [h]
        @test table[x] == ["hello, world", 1234]
    end

    @testset "Can reset the table with reset!" begin
        hashfn = SimHash(64)
        table = LSHTable(hashfn; unique_values=true)

        # Insert a few random vectors into the table that should have
        # different keys (since n_hashes == 64 is quite large). All
        # the values should also be unique.
        X = randn(16, 20)
        V = rand(20)

        insert!(table, X, V)
        @test length(keys(table)) == 20

        table1 = deepcopy(table)

        # Now reset! the table; it should be completely cleared out
        reset!(table)
        @test length(keys(table)) == 0

        # Now if we re-insert into the table, we should get the same
        # table from before.
        insert!(table, X, V)
        @test table == table1
    end

    @testset "Can specify value type with the 'valtype' keyword" begin
        hashfn = SimHash(5)
        table = LSHTable(hashfn; valtype=String)

        # Insertion should work correctly when the values have the correct type
        x = rand(16)
        insert!(table, x, "test")
        k = deepcopy(keys(table))

        @test length(k) == 1
        @test table[x] == ["test"]

        # Insertion should fail when the values have incorrect type
        @test_throws MethodError insert!(table, x, 1234)
        @test keys(table) == k && table[x] == ["test"]
        @test_throws MethodError insert!(table, x, 'a')
        @test keys(table) == k && table[x] == ["test"]
    end

    @testset "Values are unique when unique_values == true" begin
        hashfn = SimHash(64)
        table = LSHTable(hashfn; unique_values=true)

        x = randn(16)
        insert!(table, x, "test 1")
        k1 = deepcopy(keys(table))

        @test table[x] == ["test 1"]
        @test length(k1) == 1

        # Create a random data point with a different hash
        y = randn(16)
        @test index_hash(table, y) != index_hash(table, x)

        # If we try to re-insert this value with a different key, table[x]
        # should be deleted.
        insert!(table, y, "test 1")
        k2 = deepcopy(keys(table))

        @test k2 != k1
        @test length(keys(table)) == 1
        @test table[y] == ["test 1"]

        # If we try to re-insert with the _same_ key, the table should be
        # unchanged
        insert!(table, y, "test 1")
        @test keys(table) == k2
        @test table[y] == ["test 1"]

        # If we try to insert a new value, then we shouldn't have any issues
        insert!(table, x, "test 2")
        insert!(table, y, "test 3")

        @test keys(table) == k1 ∪ k2
        @test table[x] == ["test 2"]
        @test table[y] == ["test 1", "test 3"]
    end

    @testset "Cannot perform multiple insertions over AsymmetricLSHFunctions" begin
        #=
        Some hash functions have the property that hashes are data-dependent.
        For instance, in the case of MIPSHash and SignALSH, they depend on the
        largest norm among the vectors that are inserted into the table.
        If a user inserts a new item into the table, it could incur a high
        cost to both compute and memory as we may need to re-compute previously
        inserted hashes. Thus, hash tables should only be able to be inserted
        into once for such hash functions.
        =#
        input_size = 16
        n_inputs = 128
        n_hashes = 8

        hashfn_mips = MIPSHash(n_hashes; maxnorm=input_size)
        hashfn_sign = SignALSH(n_hashes; maxnorm=input_size)

        for hashfn in (hashfn_mips, hashfn_sign)
            table = LSHTable(hashfn)

            X = rand(input_size, n_inputs)
            V = rand(n_inputs)
            insert!(table, X, V)

            # Should get an error if we try to insert new inputs into the table
            @test_throws ErrorException insert!(table, rand(input_size), rand())
            @test_throws ErrorException insert!(table, rand(input_size, n_inputs), rand(n_inputs))

            # ... however, if we reset the table, then it should be possible to
            # insert new inputs into it again.
            reset!(table)
            insert!(table, X, rand(n_inputs))

            @test_throws ErrorException insert!(table, rand(input_size), rand())
        end
    end

    @testset "Create an LSHTable that uses Sets for entries" begin
        input_size = 16
        hashfn = SimHash(64)
        table = LSHTable(hashfn; valtype=Int64, entrytype=Set)

        X = rand(input_size, 128)
        insert!(table, X, 1:size(X,2))

        @test all(isa(table[x], Set{Int64}) for x in eachcol(X))

        # Should be able to set the unique_values flag even when entrytype=Set
        table = LSHTable(hashfn; unique_values=true, entrytype=Set)
        x = randn(input_size)
        insert!(table, x, "test")

        @test length(keys(table)) == 1
        @test table[x] == Set(["test"])

        y = randn(input_size)
        insert!(table, y, "test")

        @test length(keys(table)) == 1
        @test table[y] == Set(["test"])
    end

    @testset "Get an empty set/empty vector when no collisions are found" begin
        hashfn = SimHash(5)

        table = LSHTable(hashfn; valtype=Integer, entrytype=Vector)
        @test table[rand(16)] == Vector{Integer}(undef, 0)

        table = LSHTable(hashfn; entrytype=Set{String})
        @test table[rand(16)] == Set{String}()
    end
end
