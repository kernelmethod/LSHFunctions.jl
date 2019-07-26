module LSH

using Distributions, LinearAlgebra, SparseArrays

include("utils.jl")
include("LSHBase.jl")
include("simhash.jl")
include("lphash.jl")
include("mips_hash.jl")
include("sign_alsh.jl")

export SimHash, LpHash, L1Hash, L2Hash, MIPSHash,
	SignALSH, hashtype, index_hash, query_hash, n_hashes

end # module
