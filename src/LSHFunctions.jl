module LSHFunctions

using Distributions, LinearAlgebra, SparseArrays

#========================
Common types/utilities used throughout the module
========================#

include(joinpath("utils", "hash_compression.jl"))
include(joinpath("utils", "vecops.jl"))
include("LSHBase.jl")
include("intervals.jl")
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

include(joinpath("function_hashing", "monte_carlo.jl"))

#========================
Exports
========================#

# Similarity functions, norms, inner products
export cossim, inner_prod, ℓ1, ℓ2, ℓp, L1, L2, Lp, ℓ1_norm, ℓ2_norm,
       ℓp_norm, L1_norm, L2_norm, Lp_norm, jaccard

# Hash functions
export SimHash, L1Hash, L2Hash, MIPSHash, SignALSH, MinHash,
       LSHFunction, MonteCarloHash, SymmetricLSHFunction,
       AsymmetricLSHFunction

# Helper / utility functions for LSHFunctions
export index_hash, query_hash, n_hashes, hashtype, similarity, lsh_family,
       embedded_similarity, collision_probability, @interval, HashCompressor

end # module
