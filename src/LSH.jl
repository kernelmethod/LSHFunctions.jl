module LSH

using Distributions

include("LSHBase.jl")
include("symmetric.jl")
include("asymmetric.jl")

export CosSimHash, LpDistHash, L1DistHash, L2DistHash, MIPSHash

end # module
