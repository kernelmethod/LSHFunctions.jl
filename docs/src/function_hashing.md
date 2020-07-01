# Hashing in ``L^p`` function spaces

!!! warning "Under construction"
    This section is currently being developed. If you're interested in helping write this section, feel free to [open a pull request](https://github.com/kernelmethod/LSHFunctions.jl/pulls); otherwise, please check back later.

LSHFunctions supports locality-sensitive hashing over ``L^p`` function spaces. In other words, you can hash functions like `sin`, `exp`, and `f(x) = 5x^3 - 2x^2 - 9x + 1` on a few different similarities. Here's an example using [`MonteCarloHash`](@ref) over cosine similarity:

```jldoctest; setup = :(using Random; Random.seed!(0))
julia> using LSHFunctions;

julia> μ() = 2π*rand();   # μ samples a random point from [0,2π]

julia> hashfn = MonteCarloHash(cossim, μ, 3);

julia> hashfn(x -> 5x^3 - 2x^2 - 9x + 1)
3-element BitArray{1}:
 0
 1
 1
```

LSHFunctions can hash functions in any [``L^p_{\mu}(\Omega)`` function space](https://en.wikipedia.org/wiki/Lp_space) so long as ``\Omega`` has finite volume (i.e., as long as ``\int_{\Omega} d\mu(x) < +\infty``).

## Similarity statistics in function spaces
The LSHFunctions module currently supports hashing for the following similarity statistics in function spaces.

### ``L_{\mu}^p`` distance

```math
\|f - g\|_{L_{\mu}^p} = \left(\int_{\Omega} |f(x) - g(x)|^p \hspace{0.15cm} d\mu(x)\right)^{1/p}
```

### Inner product similarity

```math
\left\langle f, g\right\rangle_{L_{\mu}^2} = \int_{\Omega} f(x)g(x) \hspace{0.15cm} d\mu(x)
```

When ``f`` and ``g`` are allowed to take on complex values, ``g(x)`` is replaced by ``\overline{g(x)}`` (the complex conjugate of ``g(x)``) in the formula above.

### Cosine similarity
```math
\text{cossim}(f,g) = \frac{\left\langle f,g\right\rangle_{L_{\mu}^2}}{\|f\|_{L_{\mu}^2} \cdot \|g\|_{L_{\mu}^2}}
```

## Function approximation-based hashing

!!! warning "API subject to change"
    The API for both [`ChebHash`](@ref) and [`MonteCarloHash`](@ref), but especially the former, is being modified very quickly. As a result, the docs below may change radically for future versions of the LSHFunctions package.

Create a hash function for cosine similarity for functions in ``L^2([-1,1])``:

```
julia> hashfn = ChebHash(cossim, 50; interval=@interval(-1 ≤ x ≤ 1));

julia> n_hashes(hashfn)
50

julia> similarity(hashfn) == cossim
true

julia> hashtype(hashfn)
Bool
```

Create a hash function for ``L^2`` distance defined over ``L^2([0,2\pi])``. Hash the functions `f(x) = cos(x)` and `f(x) = x/(2π)` using the returned [`ChebHash`](@ref):

```
julia> hashfn = ChebHash(L2, 3; interval=@interval(0 ≤ x ≤ 2π));

julia> hashfn(cos)
3-element Array{Int32,1}:
  3
 -1
 -2

julia> hashfn(x -> x/(2π))
3-element Array{Int32,1}:
 0
 1
 0
```

## Monte Carlo-based hashing

Create a hash function for cosine similarity for functions in ``L^2([-1,1])``:

```jldoctest; setup = :(using LSHFunctions)
julia> μ() = 2*rand()-1;   # μ samples a random point from [-1,1]

julia> hashfn = MonteCarloHash(cossim, μ, 50; volume=2.0);

julia> n_hashes(hashfn)
50

julia> similarity(hashfn) == cossim
true

julia> hashtype(hashfn)
Bool
```

Create a hash function for ``L^2`` distance in the function space ``L^2([0,2\pi])``. Hash the functions `f(x) = cos(x)` and `f(x) = x/(2π)` using the returned [`MonteCarloHash`](@ref).

```jldoctest; setup = :(using LSHFunctions, Random; Random.seed!(0))
julia> μ() = 2π * rand(); # μ samples a random point from [0,2π]

julia> hashfn = MonteCarloHash(L2, μ, 3; volume=2π);

julia> hashfn(cos)
3-element Array{Int32,1}:
 -1
  3
  0

julia> hashfn(x -> x/(2π))
3-element Array{Int32,1}:
 -1
 -2
 -1
```

Create a hash function with a different number of sample points.

```jldoctest; setup = :(using LSHFunctions)
julia> μ() = rand();  # μ samples a random point from [0,1]

julia> hashfn = MonteCarloHash(cossim, μ; volume=1.0, n_samples=512);

julia> length(hashfn.sample_points)
512
```

## References
- Shand, William and Becker, Stephen. *Locality-sensitive hashing in function spaces*. [arXiv:2002.03909](https://arxiv.org/abs/2002.03909).
