using LSH

#========================
Global variables
========================#

const RANDOM_SEED = 0

#========================
Helper functions
========================#

mean(x) = sum(x) / length(x)

# Function that draws two random vectors and computes their hashes with
# hashfn. Returns true if the number of collisions of individual hashes
# is within δ of the expected single-hash collision probability, and
# false otherwise.
function test_collision_probability(
        hashfn :: LSH.LSHFunction,
        δ :: AbstractFloat,
        sampler = () -> randn(4))

    similarity_fn = similarity(hashfn)
    x, y = sampler(), sampler()
    sim = similarity_fn(x,y)

    hx, hy = hashfn(x), hashfn(y)
    prob = LSH.single_hash_collision_probability(hashfn, sim)
    coll_freq = mean(hx .== hy)

    prob - δ ≤ coll_freq ≤ prob + δ
end

