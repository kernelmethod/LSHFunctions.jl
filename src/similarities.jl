#=============================================

Definitions of various similarity functions

=============================================#

using QuadGK
using LinearAlgebra: dot, norm

#====================
Definitions of built-in similarity functions
====================#

#====================
Cosine similarity
====================#

@doc raw"""
    cossim(x,y)

Computes the cosine similarity between two inputs ``x`` and ``y``. Cosine similarity is defined as

``\text{cossim}(x,y) = \frac{\left\langle x,y\right\rangle}{\|x\|\cdot\|y\|}``

where ``\left\langle\cdot,\cdot\right\rangle`` is an inner product (e.g. dot product) and ``\|\cdot\|`` is its derived norm. This is roughly interpreted as being related to the angle between the inputs ``x`` and ``y``: when ``x`` and ``y`` have low angle between them, `cossim(x,y)` is high (close to ``1``). When ``x`` and ``y`` have large angle between them, `cossim(x,y)` is low (close to ``-1``).

# Arguments
- `x` and `y`: two inputs for which `dot(x,y)`, `norm(x)`, and `norm(y)` are defined.

# Examples
```jldoctest; setup = :(using LSHFunctions)
julia> using LinearAlgebra: dot, norm;

julia> x, y = rand(4), rand(4);

julia> cossim(x,y) == dot(x,y) / (norm(x) * norm(y))
true

julia> z = rand(5);

julia> cossim(x,z)
ERROR: DimensionMismatch("dot product arguments have lengths 4 and 5")
```

See also: [`SimHash`](@ref)
"""
function cossim(x::AbstractVector, y::AbstractVector)
    norm_x = norm(x)
    norm_y = norm(y)

    if norm_x == 0 || norm_y == 0
        "x and y must be nonzero" |> ErrorException |> throw
    end

    dot(x,y) / (norm_x * norm_y)
end

function cossim(f, g, interval::LSHFunctions.RealInterval)
    norm_f = L2_norm(f, interval)
    norm_g = L2_norm(g, interval)

    if norm_f == 0 || norm_g == 0
        "f and g must be nonzero" |> ErrorException |> throw
    end

    inner_prod(f, g, interval) / (norm_f * norm_g)
end

#====================
L^p distance
====================#

@doc raw"""
    ℓp(x::AbstractVector, y::AbstractVector, p::Real=2)
    ℓ1(x::AbstractVector, y::AbstractVector)
    ℓ2(x::AbstractVector, y::AbstractVector)

Computes the ``\ell^p`` distance between a pair of vectors, given by

``\ell^p(x,y) \coloneqq \|x - y\|_p = \left(\sum_i \left|x_i - y_i\right|^p\right)^{1/p}``

`ℓ1(x,y)` is the same as `ℓp(x,y,1)`, and `ℓ2(x,y)` is the same as `ℓp(x,y,2)`.

# Examples
```jldoctest; setup = :(using LSHFunctions)
julia> x = [1, 2, 3];

julia> y = [4, 5, 6];

julia> ℓp(x,y,2) == (abs(1-4)^2 + abs(2-5)^2 + abs(3-6)^2)^(1/2)
true

julia> ℓp(x,y,3) == (abs(1-4)^3 + abs(2-5)^3 + abs(3-6)^3)^(1/3)
true
```

See also: [`ℓp_norm`](@ref), [`L1Hash`](@ref), [`L2Hash`](@ref)
"""
ℓp(x::AbstractVector, y::AbstractVector, p::Real=2) = Lp(x, y, p)

@doc (@doc ℓp)
ℓ1(x::AbstractVector, y::AbstractVector) = L1(x, y)

@doc (@doc ℓp)
ℓ2(x::AbstractVector, y::AbstractVector) = L2(x, y)

@doc raw"""
    Lp(x::AbstractVector, y::AbstractVector, p::Real=2)
    L1(x::AbstractVector, y::AbstractVector)
    L2(x::AbstractVector, y::AbstractVector)

Computes the ``ℓ^p`` distance between a pair of vectors ``x`` and ``y``. Identical to `ℓp(x,y,p)`, `ℓ1(x,y)`, and `ℓ2(x,y)`, respectively.

See also: [`ℓp`](@ref)
"""
function Lp(x::AbstractVector{T}, y::AbstractVector, p::Real=2) where T
    if p ≤ 0
        "p must be positive" |> ErrorException |> throw
    elseif length(x) != length(y)
        "length(x) != length(y)" |> DimensionMismatch |> throw
    end

    result = T(0)
    @inbounds @simd for ii = 1:length(x)
        result += abs(x[ii] - y[ii])^p
    end

    return result^(1/p)
end

@doc (@doc Lp)
function L1(x::AbstractVector{T}, y::AbstractVector) where T
    if length(x) != length(y)
        "length(x) != length(y)" |> DimensionMismatch |> throw
    end

    result = T(0)
    @inbounds @simd for ii = 1:length(x)
        result += abs(x[ii] - y[ii])
    end

    return result
end

@doc (@doc Lp)
function L2(x::AbstractVector{T}, y::AbstractVector) where T
    if length(x) != length(y)
        "length(x) != length(y)" |> DimensionMismatch |> throw
    end

    result = T(0)
    @inbounds @simd for ii = 1:length(x)
        result += abs2(x[ii] - y[ii])
    end

    return √result
end

# Function space L^p distances

@doc raw"""
    Lp(f, g, interval::LSHFunctions.RealInterval, p)
    L1(f, g, interval::LSHFunctions.RealInterval)
    L2(f, g, interval::LSHFunctions.RealInterval)

Computes the ``L^p`` distance between two functions, given by

``L^p(f,g) \coloneqq \|f - g\|_p = \left(\int_a^b \left|f(x) - g(x)\right|^p \hspace{0.15cm} dx\right)^{1/p}``

# Examples
Below we compute the ``L^1``, ``L^2``, and ``L^3`` distances between ``f(x) = x^2 + 1`` and ``g(x) = 2x`` over the interval ``[0,1]``. The distances are computed by evaluating the integral

``\left(\int_0^1 \left|f(x) - g(x)\right|^p \hspace{0.15cm}dx\right)^{1/p} = \left(\int_0^1 \left|x^2 - 2x + 1\right|^p \hspace{0.15cm}dx\right)^{1/p} = \left(\int_0^1 (x - 1)^{2p} \hspace{0.15cm}dx\right)^{1/p}``

for ``p = 1``, ``p = 2``, and ``p = 3``.

```jldoctest; setup = :(using LSHFunctions)
julia> f(x) = x^2 + 1; g(x) = 2x;

julia> interval = @interval(0 ≤ x ≤ 1);

julia> Lp(f, g, interval, 1) ≈ L1(f, g, interval) ≈ 3^(-1)
true

julia> Lp(f, g, interval, 2) ≈ L2(f, g, interval) ≈ 5^(-1/2)
true

julia> Lp(f, g, interval, 3) ≈ 7^(-1/3)
true
```

See also: [`Lp_norm`](@ref), [`ℓp`](@ref)
"""
Lp(f, g, interval::LSHFunctions.RealInterval, p::Real=2) =
    Lp_norm(x -> f(x) - g(x), interval, p)

@doc (@doc Lp)
L1(f, g, interval::LSHFunctions.RealInterval) = L1_norm(x -> f(x) - g(x), interval)

@doc (@doc Lp)
L2(f, g, interval::LSHFunctions.RealInterval) = L2_norm(x -> f(x) - g(x), interval)

#====================
Jaccard similarity
====================#

@doc raw"""
    jaccard(A::Set, B::Set) :: Float64

Computes the Jaccard similarity between sets ``A`` and ``B``, which is defined as

``\text{Jaccard}(A,B) = \frac{\left|A \cap B\right|}{\left|A \cup B\right|}``

# Arguments
- `A::Set`, `B::Set`: two sets whose Jaccard similarity we would like to compute.

# Examples
```jldoctest; setup = :(using LSHFunctions)
julia> A, B = Set([1, 2, 3]), Set([2, 3, 4]);

julia> jaccard(A,B)
0.5

julia> jaccard(A,B) == length(A ∩ B) / length(A ∪ B)
true
```

See also: [`MinHash`](@ref)
"""
function jaccard(A::Set, B::Set) :: Float64
    if isempty(A)
        # Use the convention that if A = B = ∅, their Jaccard
        # similarity is zero.
        Float64(0)
    else
        length(A ∩ B) / length(A ∪ B)
    end
end

@doc raw"""
    function jaccard(x::BitArray{1}, y::BitArray{1})

Computes the Jaccard similarity between a pair of binary vectors:

``J(x, y) = \frac{\sum_{i} \min{(x_i,y_i)}}{\sum_{i} \max{(x_i,y_i)}}``

# Arguments
- `x::BitArray{1}`, `y::BitArray{1}`: two binary vectors, in the form of `BitArray`s.

# Examples
```jldoctest; setup = :(using LSHFunctions)
julia> x = BitArray([true, false, true, true, false]);

julia> y = BitArray([false, false, true, true, true]);

julia> jaccard(x,y)
0.5
```
"""
function jaccard(x::BitArray{1}, y::BitArray{1}) :: Float64
    union = sum(x .| y)
    if union == 0
        # Use the convention that if x and y are full of zeros, their Jaccard
        # similarity is zero.
        Float64(0)
    else
        intersection = sum(x .& y)
        intersection / union
    end
end

@doc raw"""
    function jaccard(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})

Computes the Jaccard similarity between a pair of vectors of real numbers:

``J(x, y) = \frac{\sum_{i} \min{(x_i,y_i)}}{\sum_{i} \max{(x_i,y_i)}}``

# Arguments
- `x::AbstractVector{<:Real}`, `y::AbstractVector{<:Real}`: a pair of vectors containing real numbers (subtypes of `Real`).

# Examples
```jldoctest; setup = :(using LSHFunctions)
julia> x = [0.8, 0.1, 0.3, 0.4, 0.1];

julia> y = [1.0, 0.6, 0.0, 0.4, 0.5];

julia> jaccard(x,y)
0.5
```
"""
function jaccard(x::AbstractVector{T},
                 y::AbstractVector) :: Float64 where {T <: Real}
    if length(x) != length(y)
        DimensionMismatch("dimensions must match") |> throw
    end

    intersection = T(0)
    union = T(0)

    @inbounds @simd for ii = 1:length(x)
        if 0 ≤ x[ii] ≤ y[ii]
            intersection += x[ii]
            union += y[ii]
        elseif 0 ≤ y[ii] < x[ii]
            intersection += y[ii]
            union += x[ii]
        else
            ErrorException("vectors must have non-negative elements") |> throw
        end
    end

    if union == T(0)
        # Use the convention that if x and y are full of zeros, their Jaccard
        # similarity is zero.
        Float64(0)
    else
        Float64(intersection / union)
    end
end

jaccard(x::AbstractVector{<:Integer}, y::AbstractVector{<:AbstractFloat}) =
    jaccard(y, x)

@doc raw"""
    function jaccard(A::Set{<:K},
                     B::Set{<:K},
                     weights::Dict{K,V}) where {K,V<:Number}

Computes the weighted Jaccard similarity between two sets:

``J(x, y) = \frac{\sum_{x\in A\cap B} w_x}{\sum_{y\in A\cup B} w_y}``

# Arguments
- `A::Set`, `B::Set`: two sets whose Jaccard similarity we would like to compute.
- `weights::Dict`: a dictionary mapping symbols in the sets `A` and `B` to numerical weights. These weights must be positive.

# Examples
```jldoctest; setup = :(using LSHFunctions)
julia> A = Set(["a", "b", "c"]);

julia> B = Set(["b", "c", "d"]);

julia> W = Dict("a" => 0.2, "b" => 2.4, "c" => 0.6, "d" => 1.8);

julia> jaccard(A,B,W)
0.6
```
"""
function jaccard(A::Set{<:K},
                 B::Set{<:K}, 
                 weights::Dict{K,V}) :: Float64 where {K,V<:Real}

    union_weight = V(0)

    for el in A ∪ B
        w = weights[el]
        if w < 0
            ErrorException("weights must be non-negative") |> throw
        end
        union_weight += w
    end

    intersection_weight = sum(weights[el] for el in A ∩ B)

    # By convention, if A = B = ∅, their Jaccard similarity is zero
    if union_weight == V(0)
        Float64(0)
    else
        Float64(intersection_weight / union_weight)
    end
end

#====================
Inner product and norms
====================#

### Inner products

@doc raw"""
    inner_prod(x::AbstractVector, y::AbstractVector)

Computes the ``\ell^2`` inner product (dot product)

``\left\langle x, y\right\rangle = \sum_i x_iy_i``

# Examples
```jldoctest; setup = :(using LSHFunctions)
julia> using LinearAlgebra: dot;

julia> x, y = randn(4), randn(4);

julia> inner_prod(x,y) ≈ dot(x,y)
true
```
"""
inner_prod(x::AbstractVector, y::AbstractVector) = dot(x,y)

# 1-dimensional inner product between L^2 functions
@doc raw"""
    inner_prod(f, g, interval::LSHFunctions.RealInterval)

Computes the ``L^2`` inner product

``\left\langle f, g\right\rangle = \int_a^b f(x)g(x) \hspace{0.15cm} dx``

where the interval we're integrating over is specified by the `interval` argument.

# Examples
```jldoctest; setup = :(using LSHFunctions)
julia> f(x) = cos(x); g(x) = sin(x);

julia> inner_prod(f, g, @interval(0 ≤ x ≤ π/2)) ≈ 1/2
true
```
"""
inner_prod(f, g, interval::LSHFunctions.RealInterval) =
    quadgk(x -> f(x)g(x), interval.lower, interval.upper)[1]

### L^p norms
@doc raw"""
    Lp_norm(x::AbstractVector, p::Real = 2)
    L1_norm(x::AbstractVector)
    L2_norm(x::AbstractVector)

Compute the ``\ell^p`` norm of a vector ``x``. Identical to `ℓp_norm(x,p)`, `ℓ1_norm(x)`, and `ℓ2_norm(x)`, respectively.

See also: [`ℓp_norm`](@ref)
"""
Lp_norm(x::AbstractVector, p::Real = 2) = norm(x,p)

@doc (@doc Lp_norm)
L1_norm(x::AbstractVector) = norm(x,1)

@doc (@doc Lp_norm)
L2_norm(x::AbstractVector) = norm(x)

@doc raw"""
    ℓp_norm(x::AbstractVector, p::Real = 2)
    ℓ1_norm(x::AbstractVector)
    ℓ2_norm(x::AbstractVector)

Compute the ``\ell^p`` norm of a point ``x``, defined as

``\|x\|_p = \left(\sum_i \left|x_i\right|^p\right)^{1/p}``

# Examples

```jldoctest; setup = :(using LSHFunctions)
julia> x = randn(4);

julia> ℓp_norm(x, 1) ≈ ℓ1_norm(x) ≈ (map(u -> abs(u)^1, x) |> sum)^(1/1)
true

julia> ℓp_norm(x, 2) ≈ ℓ2_norm(x) ≈ (map(u -> abs(u)^2, x) |> sum)^(1/2)
true

julia> ℓp_norm(x, 3) ≈ (map(u -> abs(u)^3, x) |> sum)^(1/3)
true
```

See also: [`ℓp`](@ref), [`Lp_norm`](@ref)
"""
ℓp_norm(x::AbstractVector, p::Real = 2) = Lp_norm(x, p)

@doc (@doc ℓp_norm)
ℓ1_norm(x::AbstractVector) = L1_norm(x)

@doc (@doc ℓp_norm)
ℓ2_norm(x::AbstractVector) = L2_norm(x)

# 1-dimensional L^p norms

@doc raw"""
    Lp_norm(f, interval::LSHFunctions.RealInterval, p::Real=2)
    L1_norm(f, interval::LSHFunctions.RealInterval)
    L2_norm(f, interval::LSHFunctions.RealInterval)

Computes the ``L^p`` function-space norm of a function ``f``, which is given by the equation

``\|f\|_p = \left(\int_a^b \left|f(x)\right|^p \hspace{0.15cm} dx\right)^{1/p}``
        
`L1_norm(f, interval)` is the same as `Lp_norm(f, interval, 1)`, and `L2_norm(f, interval)` is the same as `Lp_norm(f, interval, 2)`.

# Examples

```jldoctest; setup = :(using LSHFunctions)
julia> f(x) = x;

julia> interval = @interval(0 ≤ x ≤ 1);

julia> Lp_norm(f, interval, 1) ≈ L1_norm(f, interval) ≈ 2^(-1/1)
true

julia> Lp_norm(f, interval, 2) ≈ L2_norm(f, interval) ≈ 3^(-1/2)
true

julia> Lp_norm(f, interval, 3) ≈ 4^(-1/3)
true
```
"""
Lp_norm(f, interval::LSHFunctions.RealInterval, p::Real=2) = (quadgk(x -> abs(f(x)).^p, interval.lower, interval.upper)[1])^(1/p)

@doc (@doc Lp_norm)
L1_norm(f, interval::LSHFunctions.RealInterval) = quadgk(x -> abs(f(x)), interval.lower, interval.upper)[1]

@doc (@doc Lp_norm)
L2_norm(f, interval::LSHFunctions.RealInterval) = √quadgk(x -> abs2(f(x)), interval.lower, interval.upper)[1]
