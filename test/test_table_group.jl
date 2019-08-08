#=
Tests for LSHTableGroup in src/tables/table_group.jl
=#

using Test, Random, LSH

@testset "LSHTableGroup tests" begin
	Random.seed!(0)

	@testset "Create an LSHTableGroup over SimHash" begin
		input_size = 16
		n_hashes = 64
		n_tables = 3

		hashfn = SimHash(input_size, n_hashes)
		tables = LSHTableGroup(hashfn, n_tables; valtype=String)

		# Insert a vector x into the table, and then retrieve it
		x = rand(input_size)
		insert!(tables, x, "test")
		@test isa(tables[x], Set{String})
		@test tables[x] == Set(["test"])

		# All of the hash functions should be different, hence all of the hash
		# tables should be different too
		@test let result = true
			for ii = 1:n_tables
				for jj = 1:n_tables
					if ii == jj
						continue
					else
						result &= (tables.tables[ii] != tables.tables[jj])
					end
				end
			end
			result
		end

		# Should get an empty set if we query for a vector without any collisions
		y = rand(input_size)
		@test tables[y] == Set{String}()
	end

	@testset "LSHTableGroup takes the union of collisions in each table" begin
		# Create an LSHTableGroup over a large number of tables. It should be
		# effectively impossible for an input to get a collision in _every_
		# table, but extremely likely for it to get a collision in _at least one_
		# table.
		input_size = 4
		n_hashes = 2
		n_tables = 256
		n_inputs = 32

		tables = LSHTableGroup(() -> SimHash(input_size, n_hashes), n_tables; valtype=Int64)

		X = randn(input_size, n_inputs)
		insert!(tables, X, 1:n_inputs)

		# Even a random vector should get a collision with every vector in X
		y = randn(input_size)
		@test tables[y] |> collect |> sort! == 1:n_inputs
	end
end
