using LSH

#========================
Global variables
========================#

const RANDOM_SEED = 0

#========================
Helper types
========================#

# ShiftedSine so that we can quickly generate functions of the form
# f(x) = sin(αx+δ) without having to use closures, which can be relatively more
# time-intensive.
struct ShiftedSine{S <: Real, T <: Real}
    α :: T
    δ :: T
end

ShiftedSine(α::S, δ::T) where {S,T} = ShiftedSine{S,T}(α,δ)

(f::ShiftedSine)(x) = sin(f.α * x + f.δ)

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

"""
    create_step_function(N::Integer; step_generator = i -> randn())

Create a random function on the interval `[0,N]` that is piecewise constant over the intervals `[i,i+1]`, `i = 1, ..., N`. These functions have the nice property that you can isomorphically embed them in `R^N` by mapping a function to its length-`N` list of steps.

# Arguments
- `N::Integer`: the number of steps that the function should take.

# Keyword parameters
- `step_generator`: a function that takes a step number `i` as input and returns a value for the function at the `i`th step. Note that the steps start from index `1`.

# Returns
Returns the step function as well as the `N` steps.
"""
function create_step_function(N::Integer; step_generator = i -> randn())
    steps = map(step_generator, 1:N)

    return (x -> _step_function(x, steps), steps)
end

function _step_function(x, steps)
    @assert 0 ≤ x ≤ length(steps)
    current_step  = Int64(ceil(x))
    steps[current_step]
end
