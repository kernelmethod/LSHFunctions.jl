using Pkg

Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()
Pkg.activate(); Pkg.instantiate()

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter, LSH

makedocs(
    sitename = "LSH.jl",
    format   = Documenter.HTML(),
    modules  = [LSH],
    pages    = ["Home" => "index.md",
                "Similarity functions" => [
                    "Cosine similarity" => joinpath("similarities", "cosine.md"),
                    raw"``\ell^p`` distance" => joinpath("similarities", "lp_distance.md"),
                    "Jaccard similarity" => joinpath("similarities", "jaccard.md"),
                    "Inner product similarity" => joinpath("similarities", "inner_prod.md")]
               ]
)

deploydocs(
    repo = "github.com/kernelmethod/LSH.jl.git"
)
