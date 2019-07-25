module LSH

using Distributions

include("LSHBase.jl")
include("cossim_hash.jl")
include("lpdist_hash.jl")
include("mips_hash.jl")

export SimHash, LpHash, L1DistHash, L2DistHash, MIPSHash,
	hashtype, index_hash, query_hash

end # module
