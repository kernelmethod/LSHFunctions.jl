#=
Hash tables for use with LSH functions.
=#

struct LSHTable{F<:LSHFunction,H,V}
	hashfn :: F
	table :: Dict{H,Vector{V}}
end

# Outer constructors
function LSHTable(hashfn; valtype=Any)
	table = Dict{hashtype(hashfn),valtype}()
	LSHTable(hashfn, table)
end

# Access to the underlying hash function
index_hash(table::LSHTable, x) = index_hash(table.hashfn, x)
query_hash(table::LShTable, x) = query_hash(table.hashfn, x)

# Getter/setter methods
function insert(table::LSHTable, x::AbstractArray, vals)
	hashes = index_hash(table, x)

	# Hashes stored columnwise, so we iterate over them in that order
	for h in eachcol(hashes)
	end
end

insert(table::LSHTable, x::AbstractVector, v::AbstractVector) =
	insert(table, x, v[1])

function insert(lshtable::LSHTable{F,H,V}, x::AbstractVector, v) where {F,H,V}
	ih = index_hash(lshtable, x)
	entry = get!(lshtable.table, ih) do
		Vector{V}(undef, 0)
	end
	push!(entry, v)
end

# Extensions of base methods
function Base.getindex(lshtable, x) end
