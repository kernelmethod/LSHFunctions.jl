module LSH

using Distributions

include("LSHBase.jl")
include("cossim_hash.jl")
include("lpdist_hash.jl")
include("mips_hash.jl")

export CosSimHash, LpDistHash, L1DistHash, L2DistHash, MIPSHash, MIPSHash_P_LSH,
	MIPSHash_Q_LSH

end # module
