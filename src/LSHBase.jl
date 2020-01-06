#================================================================

Common typedefs and functions used throughout the LSH module.

================================================================#

#========================
Abstract typedefs
========================#
abstract type LSHFunction end
abstract type SymmetricLSHFunction <: LSHFunction end
abstract type AsymmetricLSHFunction <: LSHFunction end

#========================
LSHFunction API
========================#

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
