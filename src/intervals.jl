#=============================================

Defines a RealInterval type, used to represent intervals of the real line.
RealIntervals are primarily used for representing the output ranges of similarity
functions.

=============================================#

#====================
RealInterval definition and constructors
====================#

@doc """
    struct RealInterval{T<:Real}

Encodes an interval of the real line, such as `[-1,1]` or `[0,Inf)`.

# Fields
- `lower::T`: lower bound on the interval.
- `upper::T`: upper bound on the interval.
- `closed_below::Bool`: whether or not the interval is closed below.
- `closed_above::Bool`: whether or not the interval is closed above.

# Examples
The following snippet constructs `RealInterval` represeting the interval [0,1)

```jldoctest; setup = :(using LSHFunctions)
julia> interval = LSHFunctions.RealInterval(0, 1, true, false);
```

It's generally easier to construct an interval using the `@interval` macro. Check out the documentation for [`@interval`](@ref) for more information.

See also: [`@interval`](@ref)
"""
struct RealInterval{T<:Real}
    # Lower bound on the interval
    lower :: T

    # Upper bound on the interval
    upper :: T

    # Whether or not the lower end of the interval is closed
    closed_below :: Bool

    # Whether or not the upper end of the interval is closed
    closed_above :: Bool

    ### Internal RealInterval constructors
    RealInterval{T}(L, U, CB, CA) where T = RealInterval{T}(T(L), T(U), CB, CA)
    RealInterval{T}(L::T, U::T, CB::Bool, CA::Bool) where T =
        new{T}(L, U, CB, CA)
end

### External RealInterval constructors
@generated function RealInterval(L::Real, U::Real, CB::Bool, CA::Bool)
    bounds_type = begin
        if L <: Integer && U <: AbstractFloat
            U
        else
            L
        end
    end
    return :(RealInterval{$bounds_type}(L, U, CB, CA))
end

### String conversion and displaying RealInterval structs
function Base.string(interval::RealInterval)
    lower_bracket = (interval.closed_below) ? "[" : "("
    upper_bracket = (interval.closed_above) ? "]" : ")"
    lower_bracket * string(interval.lower) * "," * string(interval.upper) * upper_bracket
end

show(io::IO, interval::RealInterval) = print(io, string(interval))

#====================
Operator definitions for RealInterval
====================#
# Interval membership
function Base.:(∈)(x::Real, interval::RealInterval)
    if interval.lower < x < interval.upper
        true
    elseif interval.lower == x && interval.closed_below
        true
    elseif interval.upper == x && interval.closed_above
        true
    else
        false
    end
end

# Interval intersection
function Base.:(∩)(int1::RealInterval, int2::RealInterval)
    lower = max(int1.lower, int2.lower)
    upper = min(int1.upper, int2.upper)
    closed_below = (lower ∈ int1) && (lower ∈ int2)
    closed_above = (upper ∈ int1) && (upper ∈ int2)
    RealInterval(lower, upper, closed_below, closed_above)
end

intersect(int1, int2) = int1 ∩ int2

# Equality
# There are two cases in which equality can hold:
# - both intervals are empty; or
# - both intervals are non-empty, in which case the intervals can only be
#   equal if all of their fields are equal.
function Base.:(==)(int1::RealInterval, int2::RealInterval)
    # Case 1
    if isempty(int1)
        isempty(int2)
    # Case 2
    else
        int1 === int2
    end
end

#====================
Definitions of other methods for RealInterval
====================#

@doc """
    isempty(interval::RealInterval)

Returns `true` if `interval` is empty (i.e. there doesn't exist an `x` for which `x ∈ interval` is `true`), and `false` otherwise.
"""
Base.isempty(interval::RealInterval) =
    interval.lower > interval.upper ||
    (interval.lower == interval.upper && !(interval.closed_below && interval.closed_above))

@doc """
    width(interval::RealInterval)

Return the width of a `RealInterval` (i.e. the difference between its upper and lower bounds.
"""
width(interval::RealInterval{T}) where T =
    isempty(interval) ? T(0) : (interval.upper - interval.lower)

#====================
The @interval macro, which provides the primary interface for users to construct
new RealIntervals.
====================#

@doc """
    @interval(expr)

Construct a new `LSHFunctions.RealInterval` representing an interval on the real line from an expression such as

    0 ≤ x < 1

The returned expression constructs an `LSHFunctions.RealInterval` encoding the lower and upper bounds on the interval, as well as whether the ends are opened or closed.

# Examples
You can construct an interval using the following syntax:

```jldoctest; setup = :(using LSHFunctions)
julia> interval = @interval(0 ≤ x < 1);
```

There are usually multiple ways of constructing the same interval. For instance, each of the expressions below are equivalent ways of constructing the interval `[-1,1]`.

```jldoctest; setup = :(using LSHFunctions)
julia> @interval(-1 ≤  x ≤  1) ==
       @interval(-1 <= x <= 1) ==
       @interval(-1 ≤  y ≤  1) ==
       @interval( 1 ≥  x ≥ -1)
true
```

You can even create intervals with `Inf` or `-Inf` at the endpoints, e.g. `@interval(-Inf < x < Inf)`.

There are two primary operations you can run on an interval: testing for membership and intersection. You can test whether or not `x` is in an interval using `x ∈ interval`, as shown below.

```jldoctest; setup = :(using LSHFunctions)
julia> interval = @interval(0 ≤ x < 1);

julia> 0 ∈ interval && 1 ∉ interval
true

julia> 0 in interval    # This syntax also works
true
```

You can also intersect two intervals using the `∩` operator (or by using `intersect(interval_1, interval_2)`).

```
julia> @interval(0 ≤ x < 1) ∩ @interval(1/2 < x ≤ 1) == @interval(1/2 < x < 1)
true
```

See also: [`RealInterval`](@ref)
"""
macro interval(expr)
    invalid_expr_msg = """
    Expression is invalid. The expression passed to @interval must take the form

        a {<,≤,>,≥} x {<,≤,>,≥} b

    where a and b evaluate to real numbers, and x is a valid variable identifier.
    See the documentation for @interval for more information.
    """

    if !isa(expr, Expr)
        return :($invalid_expr_msg |> ErrorException |> throw)
    end

    args = expr.args

    # Convert >= and <= to ≥ and ≤ to reduce the number of symbols we have to deal with
    map!(sym -> (sym == :>=) ? :≥ : sym, args, args)
    map!(sym -> (sym == :<=) ? :≤ : sym, args, args)

    valid_expr = (
        length(args) == 5          &&
        expr.head == :comparison   &&

        # 3rd argument must be a valid variable identifier
        isa(args[3], Symbol)       &&
        Base.isidentifier(args[3]) &&

        # 2nd and 4th arguments must be inequalities
        args[2] ∈ (:<, :≤, :>, :≥) &&
        args[4] ∈ (:<, :≤, :>, :≥)
    )

    if !valid_expr
        return :($invalid_expr_msg |> ErrorException |> throw)
    end

    # Extract a tuple (a, op1, b, op2) from the expression $a $op1 x $op2 $b. E.g.
    # if expr == :(0 ≤ x < b), return (0, :≤, :b, :<)
    interval_tuple = begin
        # Flip inequalities if expression is of the form b {>,≥} x {>,≥} a
        if args[2] in (:>, :≥) && args[4] in (:>, :≥)
            map!(x -> (x == :>) ? :< : x, args, args)
            map!(x -> (x == :≥) ? :≤ : x, args, args)
            reverse!(args)
        elseif !(args[2] in (:<, :≤) && args[4] in (:<, :≤))
            return quote
                "Inequalities point in opposite directions" |>
                ErrorException |>
                throw
            end
        end
        (args[1], args[2], args[5], args[4])
    end

    # Construct a new RealInterval based on the inequalities that were provided
    a, b = interval_tuple[1], interval_tuple[3]
    closed_below = (interval_tuple[2] == :≤)
    closed_above = (interval_tuple[4] == :≤)
    quote
        RealInterval($(esc(a)), $(esc(b)), $(esc(closed_below)), $(esc(closed_above)))
    end
end
