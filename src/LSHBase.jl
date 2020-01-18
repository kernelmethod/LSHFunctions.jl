#================================================================

Common typedefs and functions used throughout the LSH module.

================================================================#

#========================
Global variables and constants
========================#

# available_similarities: a set of all of the similarity functions that have
# been associated with hash functions via the register_similarity! macro.
const available_similarities = Set()

# Defaults to use for common arguments
const DEFAULT_N_HASHES = 1
const DEFAULT_DTYPE = Float32
const DEFAULT_RESIZE_POW2 = false

#========================
Abstract typedefs
========================#

abstract type LSHFunction end
abstract type SymmetricLSHFunction <: LSHFunction end
abstract type AsymmetricLSHFunction <: LSHFunction end

#========================
Similarity function API
========================#

# Value type to encode different similarity functions
struct SimilarityFunction{F} end
SimilarityFunction(sim) = SimilarityFunction{sim}()

#========================
LSHFunction API
========================#

macro register_similarity! end
function lsh_family end

#=
The following functions must be defined for all LSHFunction subtypes
=#
function hashtype end
function n_hashes end
function similarity end

#========================
SymmetricLSHFunction API
========================#

index_hash(h :: SymmetricLSHFunction, x) = h(x)
query_hash(h :: SymmetricLSHFunction, x) = h(x)

#=
The following functions must be defined for all SymmetricLSHFunction subtypes
=#
function (:: SymmetricLSHFunction) end

#========================
Abstract typedefs
========================#

#=
The following functions must be defined for all AsymmetricLSHFunction subtypes
=#
function index_hash end
function query_hash end

#========================
Documentation utilities
========================#

available_similarities_as_strings() = available_similarities .|> string |> sort

### Docstring generators for common keyword arguments
N_HASHES_DOCSTR(; default = DEFAULT_N_HASHES) = """
`n_hashes::Integer` (default: `$(default)`): the number of hash functions to generate."""

DTYPE_DOCSTR(hashfn; default = DEFAULT_DTYPE) = """
`dtype::DataType` (default: `$(default)`): the data type to use in the $(hashfn) internals. For performance reasons you should pick `dtype` to match the type of the data you're hashing."""

RESIZE_POW2_DOCSTR(hashfn; default = DEFAULT_RESIZE_POW2) = """
`resize_pow2::Bool` (default: `$(default)`): affects the way in which the returned `$(hashfn)` resizes to hash inputs of different sizes. If you think you'll be hashing inputs of many different sizes, it's more efficient to set `resize_pow2 = true`."""

