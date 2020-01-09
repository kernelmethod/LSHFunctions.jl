module LSH

using Distributions, LinearAlgebra, SparseArrays

include("utils.jl")
include("LSHBase.jl")

include("similarities.jl")

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

# Similarity functions
export cossim, ℓ_1, ℓ_2, ℓ_p, jaccard

# Hash functions
export SimHash, L1Hash, L2Hash, MIPSHash, SignALSH, MinHash,
       LSHFunction

# Helper / utility functions for LSHFunctions
export index_hash, query_hash, n_hashes, hashtype, similarity

# Hash tables and related functions
export LSHTable, LSHTableGroup, insert!, reset!

end # module
