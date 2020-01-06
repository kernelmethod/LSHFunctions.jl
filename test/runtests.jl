#================================================================

Testrunner for the LSH module

================================================================#

#========================
Global variables
========================#

const RANDOM_SEED = 0

#========================
Helper functions
========================#

mean(x) = sum(x) / length(x)

#========================
Tests
========================#

include("test_simhash.jl")
include("test_minhash.jl")
include("test_lphash.jl")
include("test_mips_hash.jl")
include("test_sign_alsh.jl")
include("test_table.jl")
include("test_table_group.jl")
