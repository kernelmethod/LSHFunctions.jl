# API reference

## LSHFunction API

```@docs
LSHFunction
lsh_family
hashtype
n_hashes
similarity
collision_probability
index_hash
query_hash
SymmetricLSHFunction
AsymmetricLSHFunction
```

## Hash functions

```@docs
SimHash
MinHash
L1Hash
L2Hash
SignALSH
MIPSHash
```

## Similarity functions

```@autodocs
Modules = [LSHFunctions]
Private = false
Pages = ["similarities.jl"]
```

## Miscellaneous

```@docs
@interval
```

## Private interface

```@autodocs
Modules = [LSHFunctions]
Public = false
```
