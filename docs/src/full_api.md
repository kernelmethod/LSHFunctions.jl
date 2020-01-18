# API reference

## LSHFunction API

```@docs
LSHFunction
lsh_family
hashtype
n_hashes
similarity
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
Modules = [LSH]
Private = false
Pages = ["similarities.jl"]
```

## Private interface

```@autodocs
Modules = [LSH]
Public = false
```
