#=
Common definitions used throughout the project.
=#

#=
Global constants
=#
const LSH_FAMILY_DTYPES = Union{Float32,Float64}

#=
Abstract typedefs
=#
abstract type LSHFamily{T<:LSH_FAMILY_DTYPES} end
abstract type SymmetricLSHFamily{T} <: LSHFamily{T} end
abstract type AsymmetricLSHFamily{T} <: LSHFamily{T} end
