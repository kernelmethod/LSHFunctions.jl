module LSH

using Distributions, LinearAlgebra, SparseArrays

include("utils.jl")
include("LSHBase.jl")

#=
Hash functions
=#
include(joinpath("hashes", "simhash.jl"))
include(joinpath("hashes", "minhash.jl"))
include(joinpath("hashes", "lphash.jl"))
include(joinpath("hashes", "mips_hash.jl"))
include(joinpath("hashes", "sign_alsh.jl"))

#=
Hash tables for LSHFunctions
=#
include(joinpath("tables", "table.jl"))
include(joinpath("tables", "table_group.jl"))

#=
Exports
=#
# Hash functions
export SimHash, LpHash, L1Hash, L2Hash, MIPSHash,
       SignALSH, MinHash

# Helper / utility functions for LSHFunctions
export hashtype, index_hash, query_hash, n_hashes,
       redraw!

# Hash tables and related functions
export LSHTable, LSHTableGroup, insert!, reset!

end # module
