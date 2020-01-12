module LSH

using Distributions, LinearAlgebra, SparseArrays

#========================
Common types/utilities used through the LSH module
========================#

include("utils.jl")
include("LSHBase.jl")
include("similarities.jl")

#========================
Hash functions
========================#

include(joinpath("hashes", "simhash.jl"))
include(joinpath("hashes", "minhash.jl"))
include(joinpath("hashes", "lphash.jl"))
include(joinpath("hashes", "mips_hash.jl"))
include(joinpath("hashes", "sign_alsh.jl"))

# Must be placed last, since it uses the definitions of LSHFunction subtypes
# defined in the other files.
include(joinpath("hashes", "lshfunction.jl"))

#========================
Function hashing
========================#

include(joinpath("function_hashing", "chebhash.jl"))
include(joinpath("function_hashing", "monte_carlo.jl"))

#========================
Hash tables for LSHFunctions
========================#

include(joinpath("tables", "table.jl"))
include(joinpath("tables", "table_group.jl"))

#========================
Hash tables for LSHFunctions
========================#

# Similarity functions
export cossim, ℓ_1, ℓ_2, ℓ_p, jaccard

# Hash functions
export SimHash, L1Hash, L2Hash, MIPSHash, SignALSH, MinHash,
       LSHFunction, MonteCarloHash, ChebHash

# Helper / utility functions for LSHFunctions
export index_hash, query_hash, n_hashes, hashtype, similarity,
       embedded_similarity

# Hash tables and related functions
export LSHTable, LSHTableGroup, insert!, reset!

end # module
