#================================================================

Common typedefs and functions used throughout the LSH module.

================================================================#

#========================
Global variables and constants
========================#

# available_similarities: a set of all of the similarity functions that have
# been associated with hash functions via the register_similarity! macro.
const available_similarities = Set()

available_similarities_as_strings() = available_similarities .|> string |> sort

#========================
Abstract typedefs
========================#
abstract type LSHFunction end
abstract type SymmetricLSHFunction <: LSHFunction end
abstract type AsymmetricLSHFunction <: LSHFunction end

#========================
LSHFunction API
========================#

macro register_similarity! end

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
