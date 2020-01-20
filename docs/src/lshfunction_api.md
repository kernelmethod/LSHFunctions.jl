# The LSHFunction API

!!! warning "Under construction"
    This section is currently being developed. If you're interested in helping write this section, feel free to [open a pull request](https://github.com/kernelmethod/LSHFunctions.jl/pulls); otherwise, please check back later.

## LSHFunction
The `LSH` module exposes a relatively easy interface for constructing new hash functions. Namely, you call [`LSHFunction`](@ref) with 

- the similarity function you want to use;
- the number of hash functions you want to generate; and
- keyword parameters specific to the LSH function family that you're sampling from.

```
LSHFunction(similarity, n_hashes::Integer=1; kws...)
```

For instance, in the snippet below we create a single hash function corresponding to cosine similarity:

```jldoctest
julia> using LSH

julia> hashfn = LSHFunction(cossim);

julia> typeof(hashfn)
SimHash{Float32}

julia> n_hashes(hashfn)
1

julia> similarity(hashfn)
cossim (generic function with 2 methods)
```

As another example, following code snippet creates 10 hash functions for inner product similarity. All of the generated hash functions are bundled together into a single [`SignALSH`](@ref) struct. We specify the following keyword arguments:

- `dtype`: the data type to use internally in the [`SignALSH`](@ref) struct.
- `maxnorm`: an upper bound on the norm of the data points we're hashing, and a required parameter for [`SignALSH`](@ref).

```jldoctest
julia> using LSH

julia> hashfn = LSHFunction(inner_prod, 10; dtype=Float64, maxnorm=5.0);

julia> n_hashes(hashfn)
10

julia> typeof(hashfn)
SignALSH{Float64}

julia> hashfn.maxnorm
5.0
```

!!! info "Creating multiple hash functions"
    In practice, you usually want to use multiple hash functions at the same time, and combine their hashes together in order to form a key with which to index into the hash table. To create `N` hash functions simultaneously, run 
    
    ```julia
    hashfn = LSHFunction(similarity, N; kws...)
    ```

    `hashfn` will automatically generate and compute `N` different hash functions. It will then return a `Vector` of those hashes (unless `hashtype(hashfn)` is `Bool`, in which case it will return a `BitArray`).

    - [See the FAQ](@ref Why-do-we-compute-multiple-hashes-for-every-input?) for the reasoning behind using multiple locality-sensitive hash functions simultaneously.

If you want to know what hash function will be created for a given similarity, you can use [`lsh_family`](@ref):

```jldoctest; setup = :(using LSH)
julia> lsh_family(jaccard)
MinHash

julia> lsh_family(ℓ1)
L1Hash
```

## Utilities
LSHFunctions.jl provides a few common utility functions that you can use across [`LSHFunction`](@ref) subtypes:

- [`n_hashes`](@ref): returns the number of hash functions computed by an [`LSHFunction`](@ref).

```jldoctest; setup = :(using LSH)
julia> hashfn = LSHFunction(jaccard);

julia> n_hashes(hashfn)
1

julia> hashfn = LSHFunction(jaccard, 10);

julia> n_hashes(hashfn)
10

julia> hashes = hashfn(randn(50));

julia> length(hashes)
10
```

- [`similarity`](@ref): returns the similarity function for which the input [`LSHFunction`](@ref) is locality-sensitive:

```jldoctest; setup = :(using LSH)
julia> hashfn = LSHFunction(cossim);

julia> similarity(hashfn)
cossim (generic function with 2 methods)
```

- [`hashtype`](@ref): returns the type of hash computed by the input hash function. Note that in practice `hashfn(x)` (or [`index_hash(hashfn,x)`](@ref) and [`query_hash(hashfn,x)`](@ref) for an [`AsymmetricLSHFunction`](@ref)) will return an array of hashes, one for each hash function you generated. [`hashtype`](@ref) is the data type of each element of `hashfn(x)`.

```jldoctest; setup = :(using LSH)
julia> hashfn = LSHFunction(cossim, 5);

julia> hashtype(hashfn)
Bool

julia> hashes = hashfn(rand(100));

julia> typeof(hashes)
BitArray{1}

julia> typeof(hashes[1]) == hashtype(hashfn)
true
```

- [`collision_probability`](@ref): returns the probability of collision for two inputs with a given similarity. For instance, the probability that a single MinHash hash function causes a collision between inputs `A` and `B` is equal to [`jaccard(A,B)`](@ref jaccard):

  ```jldoctest; setup = :(using LSH)
  julia> hashfn = MinHash();

  julia> A = Set(["a", "b", "c"]);

  julia> B = Set(["b", "c", "d"]);

  julia> collision_probability(hashfn, A, B) ==
         collision_probability(hashfn, jaccard(A,B)) ==
         jaccard(A,B)
  true
  ```

  We often want to compute the probability that not just one hash collides, but that multiple hashes collide simultaneously. You can calculate this using the `n_hashes` keyword argument. If left unspecified, then [`collision_probability`](@ref) will use [`n_hashes(hashfn)`](@ref n_hashes) hash functions to compute the probability.

  ```jldoctest; setup = :(using LSH)
  julia> hashfn = MinHash(5);

  julia> A = Set(["a", "b", "c"]);

  julia> B = Set(["b", "c", "d"]);

  julia> collision_probability(hashfn, A, B) ==
         collision_probability(hashfn, A, B; n_hashes=5) ==
         collision_probability(hashfn, A, B; n_hashes=1)^5
  true

  julia> sim = jaccard(A,B);

  julia> collision_probability(hashfn, sim) ==
         collision_probability(hashfn, sim; n_hashes=5) ==
         collision_probability(hashfn, sim; n_hashes=1)^5
  true
  ```

