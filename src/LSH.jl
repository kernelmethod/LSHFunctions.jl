module LSH

using Distributions

include("LSHBase.jl")
include("simhash.jl")
include("lphash.jl")
include("mips_hash.jl")

export SimHash, LpHash, L1Hash, L2Hash, MIPSHash,
	hashtype, index_hash, query_hash

end # module
