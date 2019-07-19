module LSH

using Distributions

include("hashing.jl")

export CosSimHash, LpDistHash, L1DistHash, L2DistHash, MIPSHash

end # module
