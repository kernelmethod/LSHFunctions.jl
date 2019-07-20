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

#=
APIs for LSH families.
=#

# General API for all LSHFamily types
function hashtype(::LSHFamily) end

# Asymmetric LSH families
function index_hash(::AsymmetricLSHFamily, x :: AbstractArray) end
function query_hash(::AsymmetricLSHFamily, x :: AbstractArray) end
