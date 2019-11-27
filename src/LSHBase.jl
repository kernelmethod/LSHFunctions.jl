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
abstract type LSHFunction{T <: LSH_FAMILY_DTYPES} end
abstract type SymmetricLSHFunction{T} <: LSHFunction{T} end
abstract type AsymmetricLSHFunction{T} <: LSHFunction{T} end

#=
API for LSH families.
=#

# General API for all LSHFunction types
function hashtype(:: LSHFunction) end
function n_hashes(:: LSHFunction) end
function redraw!(:: F) :: F where {F <: LSHFunction} end

redraw(h::LSHFunction) =
	h |> deepcopy |> redraw!

# Symmetric LSH families
function (:: SymmetricLSHFunction)(x) end
index_hash(h :: SymmetricLSHFunction, x) = h(x)
query_hash(h :: SymmetricLSHFunction, x) = h(x)

# Asymmetric LSH families
function index_hash(:: AsymmetricLSHFunction, x) end
function query_hash(:: AsymmetricLSHFunction, x) end
