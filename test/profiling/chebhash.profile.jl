using Random, Profile
using LSH: ChebHash, cossim

include(joinpath("..", "utils.jl"))

#========================
Benchmarks / profiling
========================#

Random.seed!(0)
Profile.init(delay = 1e-3, n = 100_000)

# Pre-compile
hashfn = ChebHash(cossim, 1024)
let f = ShiftedSine(π, π * rand())
    f(randn()); f.(randn(5))
end

# Start generating hash functions and computing hashes
Profile.clear_malloc_data()

for ii = 1:10
    f = ShiftedSine(π, π * rand())
    @profile hashfn(f)
end

open("chebhash.profile.prof", "w") do f
    Profile.print(f)
    println("Results written to ", f.name)
end
