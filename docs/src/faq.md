# FAQ

## Why do we compute multiple hashes for every input?
In a traditional hash table data structure, you have a single hash function (e.g. MurmurHash) that you use to convert inputs into hashes, which you can then use to index into the table. With LSH, you randomly generate multiple hash functions from a single LSH family. To index into the hash table, you apply each of those hash functions to your input and concatenate your computed hashes together. The concatenated hashes form the key you use to index into the hash table.

The reason for computing multiple hashes is that every LSH function provides (at most) only a few bits of additional information with which to partition the input space. For example, [`SimHash`](@ref) is a single-bit hash: that is, if you create `hashfn = SimHash()`, then `hashfn(x)` can only return either `BitArray([0])` or `BitArray([1])`. If you're trying to use `hashfn` to speed up similarity search, then the hash you compute will -- *at best* -- reduce the number of points you have to search through by only 50% on average.

In fact, the situation can be much more dire than that. If your data are highly structured, it is likely that each of your hashes will place data points into a tiny handful of buckets -- even just one bucket. For instance, in the snippet below we have a dataset of 100 points that all have very high cosine similarity with one another. If we only create a single hash function when we call [`SimHash`](@ref), then it's very likely that all of the data points will have the same hash.

```jldoctest; setup = :(using LSH, Random; Random.seed!(0))
julia> hashfn = SimHash();

julia> data = ones(10, 100);  # Each column is a data point

julia> data[end,1:end] .= rand(100);  # Randomize the last dimension of each point

julia> hashes = map(x -> hashfn(x), eachcol(data));

julia> unique(hashes)
1-element Array{BitArray{1},1}:
 [0]
```

The solution to this is to generate multiple hash functions, and combine each of the hashes we compute for an input into a single key. In the snippet below, we create 20 hash functions with [`SimHash`](@ref). Each hash computed in `map(x -> hashfn(x), eachcol(data))` is a length-20 `BitArray`.


```jldoctest; setup = :(using LSH, Random; Random.seed!(0))
julia> hashfn = SimHash(20);

julia> data = ones(10,100);  # Each column is a data point

julia> data[end,1:end] .= rand(100);  # Randomize the last dimension of each point

julia> hashes = map(x -> hashfn(x), eachcol(data));

julia> unique(hashes) |> length
3

julia> for uh in unique(hashes)
           println(sum(uh == h for h in hashes))
       end
72
16
12
```

Our hash function has generated 3 unique 20-bit hashes, with 72 points sharing the first hash, 16 points sharing the second hash, and 12 points sharing the third hash. That's not a great split, but could still drastically reduce the size of the search space. For instance, the following benchmarks (on an Intel Core i7-8565U @ 1.80 GHz) suggest that the cost of computing [`SimHash`](@ref) on 10-dimensional data is about 34 times the cost of computing [`cossim`](@ref):

```
julia> using BenchmarkTools

julia> @benchmark(hashfn(x), setup=(x=rand(10)))
BenchmarkTools.Trial: 
  memory estimate:  4.66 KiB
  allocs estimate:  6
  --------------
  minimum time:     612.231 ns (0.00% GC)
  median time:      1.563 μs (0.00% GC)
  mean time:        1.728 μs (17.60% GC)
  maximum time:     24.123 μs (92.03% GC)
  --------------
  samples:          10000
  evals/sample:     169

julia> @benchmark(cossim(x,y), setup=(x=rand(10);y=rand(10)))
BenchmarkTools.Trial: 
  memory estimate:  0 bytes
  allocs estimate:  0
  --------------
  minimum time:     46.203 ns (0.00% GC)
  median time:      46.415 ns (0.00% GC)
  mean time:        47.467 ns (0.00% GC)
  maximum time:     160.076 ns (0.00% GC)
  --------------
  samples:          10000
  evals/sample:     988

julia> 1.563e-6 / 46.415e-9
33.67445868792416
```

So as long as [`SimHash`](@ref) reduces the size of the search space by 34 data points on average, it's faster than calculating the similarity between every pair of points. Even for our tiny dataset, which only had 100 points, that's still well worth it: with the 72/16/12 split that we got, [`SimHash`](@ref) reduces the number of similarities we have to calculate by ``100 - \left(\frac{72^2}{100} + \frac{16^2}{100} + \frac{12^2}{100}\right) \approx 44`` points on average.

!!! info "Improving LSH partitioning"
    LSH can be poor at partitioning your input space when data points are very similar to one another. In these cases, it may be helpful to find ways to transform your data in order to reduce their similarity.

    For instance, in the example above, we created a synthetic dataset with the following code:

    ```julia
    julia> data = ones(10,100);  # Each column is a data point

    julia> data[end,1:end] .= rand(100);  # Randomize the last dimension of each point 
    ```

    These data are, for all practical purposes, one-dimensional. Their first nine dimensions are all the same; only the last dimension provides any unique information about a given data point. As a result, a dimensionality reduction technique like principal component analysis (PCA) would have helped de-correlate the dimensions of the data and thereby reduced the cosine similarity between pairs of points.



