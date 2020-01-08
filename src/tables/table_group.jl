#=
Utility struct for creating groups of LSHTables. When querying these groups, we
take the union of the collisions detected by each of the individual tables.
=#

struct LSHTableGroup{T<:LSHTable}
	tables :: Vector{T}
end

# Outer constructors
LSHTableGroup(hfn_generator, n_tables=1; valtype=Any, unique_values=false) =
	(LSHTable(
		hfn_generator();
		entrytype=Set{valtype},
		unique_values=unique_values) for ii = 1:n_tables) |>
	collect |>
	LSHTableGroup

#=
Extensions of LSHTable methods to groups of tables
=#

# The functions insert!, reset!, and redraw! all just apply the LSHTable version
# of those functions to each of the tables, and return the LSHTableGroup.
for func in (:(Base.insert!), :(reset!), :(redraw!))
	@eval begin
		"""
			$(string($(func)))(tablegroup :: LSHTableGroup, args...; kws...)

		Apply `$(string($(func)))(::LSHTable, args...; kws...)` to each of the tables in an LSHTableGroup.
		"""
		function $(func)(tablegroup :: LSHTableGroup, args...; kws...)
			for table in tablegroup.tables
				$(func)(table, args...; kws...)
			end

			return tablegroup
		end
	end
end

Base.getindex(tablegroup :: LSHTableGroup, x :: AbstractVector) =
	union((table[x] for table in tablegroup.tables)...)
