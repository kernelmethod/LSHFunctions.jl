#================================================================

Implementations of the LSHFunction() method for constructing new hash functions.

================================================================#

using Markdown

#========================
Macros
========================#
@doc """
    register_similarity!(similarity, hashfn)

Register `hashfn` to the `LSH` module as the default locality-sensitive hash function to use for similarity function `similarity`. This makes it possible to construct a new hash function for `similarity` with `LSHFunction(similarity, args...; kws...)`.

# Arguments
- `similarity`: the similarity function to register.
- `hashfn`: the default locality-sensitive hash function that `similarity` should be associated with.

# Examples
Create a custom implementation of cosine similarity called `my_cossim`, and associate it with `SimHash`:

```jldoctest; setup = :(using LSH)
julia> using LinearAlgebra: dot, norm

julia> my_cossim(x,y) = dot(x,y) / (norm(x) * norm(y));

julia> hashfn = LSHFunction(my_cossim);
ERROR: MethodError: no method matching LSHFunction(::typeof(my_cossim))

julia> LSH.@register_similarity!(my_cossim, SimHash);

julia> hashfn = LSHFunction(my_cossim);

julia> isa(hashfn, SimHash)
true
```
"""
macro register_similarity!(similarity, hashfn)
    lshfn = :(LSH.LSHFunction)
    lshfam = :(LSH.lsh_family)

    quote
        local similarity = $(esc(similarity))

        # Check that similarity is actually callable
        if similarity |> methods |> length == 0
            "similarity must be callable" |>
            ErrorException |>
            throw
        end

        if similarity ∈ available_similarities
            "Similarity function " * string(similarity) * " has already been registered." |>
            ErrorException |>
            throw
        end

        union!(available_similarities, Set([similarity]))

        # Define LSHFunction(similarity, args...; kws...)
        $(esc(lshfn))(::typeof($(esc(similarity))), args...; kws...) =
            $(esc(hashfn))(args...; kws...)

        # Define lsh_family(similarity)
        $(esc(lshfam))(::typeof($(esc(similarity)))) = $(esc(hashfn))
    end
end

#========================
Associate similarity functions with LSHFunction subtypes
========================#

macro reset_similarities!()
    quote
        intersect!(available_similarities, Set())

        methods(LSHFunction).ms .|> Base.delete_method
        methods(lsh_family).ms  .|> Base.delete_method

        @register_similarity!(cossim, SimHash)
        @register_similarity!(ℓ1, L1Hash)
        @register_similarity!(ℓ2, L2Hash)
        @register_similarity!(jaccard, MinHash)
        @register_similarity!(inner_prod, SignALSH)
    end
end

@reset_similarities!()

#========================
Documentation for various components of the LSHFunction API
========================#

### similarity docs

Docs.getdoc(::typeof(similarity)) = Markdown.parse("""
    similarity(hashfn::LSHFunction)

Returns the similarity function that `hashfn` hashes on.

# Arguments
- `hashfn::AbstractLSHFunction`: the hash function whose similarity we would like to retrieve.

# Returns
Returns a similarity function, which is one of the following:

```
$(join(available_similarities_as_strings(), "\n"))
```

# Examples
```jldoctest; setup = :(using LSH)
julia> hashfn = LSHFunction(cossim);

julia> similarity(hashfn) == cossim
true
```
""") # similarity

### LSHFunction docs

Docs.getdoc(::typeof(LSHFunction)) = Markdown.parse("""
    LSHFunction(similarity, args...; kws...)

Construct the default `LSHFunction` subtype that corresponds to the similarity function `similarity`.

# Arguments
- `similarity`: the similarity function you want to use. Can be any of the following:

```
$(join(available_similarities_as_strings(), "\n"))
```

- `args...`: arguments to pass on to the default `LSHFunction` constructor corresponding to `similarity`.
- `kws...`: keyword parameters to pass on to the default `LSHFunction` constructor corresponding to `similarity`.

# Returns
Returns a subtype of `LSH.LSHFunction` that hashes the similarity function `similarity`.

# Examples
In the snippet below, we construct `$(lsh_family(cossim))` (the default hash function corresponding to cosine similarity) using `LSHFunction()`:

```jldoctest; setup = :(using LSH)
julia> hashfn = LSHFunction(cossim);

julia> typeof(hashfn) <: $(lsh_family(cossim)) <: LSHFunction
true
```

We can provide arguments and keyword parameters corresponding to the hash function that we construct:

```jldoctest; setup = :(using LSH)
julia> hashfn = LSHFunction(inner_prod, 100; dtype=Float64, maxnorm=10);

julia> n_hashes(hashfn) == 100 &&
       typeof(hashfn) <: SignALSH{Float64} &&
       hashfn.maxnorm == 10
true
```

See also: [`lsh_family`](@ref)
""") # LSHFunction

### lsh_family docs

@doc """
    lsh_family(similarity)

Return the default constructor or `LSHFunction` subtype used to construct a hash function for the similarity function `similarity`.

The main use of `lsh_family` is to make it easier to find the documentation for the hash function that's constructed when you call `LSHFunction`. For instance, if you want to know more about the arguments and keyword parameters that can be given to `LSHFunction(inner_prod)`, you can run

```
julia> lsh_family(inner_prod)
SignALSH

help?> SignALSH
```

# Examples

```jldoctest; setup = :(using LSH)
julia> lsh_family(cossim)
SimHash

julia> lsh_family(ℓ1)
$(
if L1Hash |> methods |> length == 1
    "L1Hash (generic function with 1 method)"
else
    "L1Hash (generic function with $(L1Hash |> methods |> length) methods)"
end
)
```

See also: [`LSHFunction`](@ref)
""" lsh_family
