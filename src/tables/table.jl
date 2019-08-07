#=
Hash tables for use with LSH functions.
=#

"""
	struct LSHTable{H,V,F<:LSHFunction}

Specialized hash table implementation for locality-sensitive hash functions. Keys are
hashes computed by an `LSHFunction`, while values are user-specified and inserted
using the `insert!` function. To find all of the collisions for an input `x`, you
can simply index into the table, e.g. using `table[x]`.
"""
struct LSHTable{H,V,F<:LSHFunction}
	hashfn :: F
	table :: Dict{H,Vector{V}}
	value_loc :: Dict{V,Pair{H,Int64}}
	unique_values :: Bool
end

# Outer constructors
function LSHTable(hashfn; valtype=Any, unique_values::Bool=false)
	htype = hashtype(hashfn)

	table = Dict{htype,Vector{valtype}}()
	value_loc = Dict{valtype,Pair{htype,Int64}}()

	LSHTable(hashfn, table, value_loc, unique_values)
end

#=
Methods for accessing the underlying hash function
=#
index_hash(table::LSHTable, x) = index_hash(table.hashfn, x)
query_hash(table::LSHTable, x) = query_hash(table.hashfn, x)

#=
Getter/setter methods
=#

# A typeunion that covers all of the hash functions that can only
# be inserted into once.
const SINGLE_INSERT_LSHFUNCTION =
	Union{
		SignALSH,
		MIPSHash
	}

function Base.insert!(lshtable::LSHTable{H,V,F}, x, vals) where
		{H, V, F<:SINGLE_INSERT_LSHFUNCTION}

	if !isempty(keys(lshtable))
		error("LSHTables using hash functions of type $(F) can ony be inserted into once. " *
			  "If you want to insert new elements into the table, you must first reset! it.")
	else
		invoke(Base.insert!, Tuple{LSHTable,typeof(x),typeof(vals)}, lshtable, x, vals)
	end
end

# Use @generated since we want the type signature to be Tuple{LSHTable,Any,Any}, but want
# different behaviors for x::AbstractArray versus x::AbstractVector
@generated function Base.insert!(lshtable::LSHTable, x::A, vals) where A
	apply_expr = if A <: AbstractVector
		quote
			insert_at_hash!(lshtable, hashes, vals)
		end
	else
		quote
			# Hashes stored columnwise, so we iterate over them in that order
			for (ii,h) in enumerate(eachcol(hashes))
				insert_at_hash!(lshtable, h, vals[ii])
			end
		end
	end

	quote
		hashes = index_hash(lshtable, x)
		$apply_expr
	end
end

function insert_at_hash!(lshtable::LSHTable{F,H,V}, ih, v) where {F,H,V}
	entry = get!(lshtable.table, ih) do
		Vector{V}(undef, 0)
	end

	if lshtable.unique_values
		# Check if the value already exists in the table. If it does and it exists
		# at a different hash, then delete the old value.
		if haskey(lshtable.value_loc, v)
			old_ih, idx = lshtable.value_loc[v]

			if old_ih == ih
				# Skip insertion into the table if the value already exists
				# at the same hash.
				return
			else
				# Remove old entry in the hash table
				old_entry = lshtable.table[old_ih]
				deleteat!(old_entry, idx)

				if length(old_entry) == 0
					delete!(lshtable.table, old_ih)
				end
			end
		end

		# Update the value_loc dictionary to reflect the new index hash
		# and position within the lshtable.table[ih] vector
		lshtable.value_loc[v] = (ih => length(entry)+1)
	end

	push!(entry, v)
end

#=
Extensions of Base methods for Dict types
=#
function Base.getindex(lshtable::LSHTable, x)
	qh = query_hash(lshtable, x)
	(lshtable.table[h] for h in eachcol(qh))
end

function Base.getindex(lshtable::LSHTable, x::AbstractVector)
	qh = query_hash(lshtable, x)
	lshtable.table[qh]
end

Base.haskey(lshtable::LSHTable, key) =
	haskey(lshtable.table, key)

Base.keys(lshtable::LSHTable) =
	keys(lshtable.table)

Base.delete!(lshtable::LSHTable, k) =
	delete!(lshtable.table, k)

Base.:(==)(lshtable_1::LSHTable, lshtable_2::LSHTable) =
	(lshtable_1.table == lshtable_2.table)

#=
Additional methods for LSHTable
=#
function reset!(lshtable::LSHTable)
	# Delete all keys from the table
	for k in keys(lshtable)
		delete!(lshtable, k)
	end

	# If the unique_values flag is set, then we also need to clear
	# out the value_loc table
	if lshtable.unique_values
		for k in keys(lshtable.value_loc)
			delete!(lshtable.value_loc, k)
		end
	end
end

function redraw!(table::LSHTable)
	redraw!(table.hashfn)
	reset!(table)
end
