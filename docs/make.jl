using Pkg

Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()
Pkg.activate(); Pkg.instantiate()

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter, LSHFunctions

makedocs(
    sitename = "LSHFunctions.jl",
    format   = Documenter.HTML(),
    modules  = [LSHFunctions],
    pages    = ["Home" => "index.md",
                "The LSHFunction API" => "lshfunction_api.md",
                "Similarity statistics" => [
                    "Cosine similarity" => joinpath("similarities", "cosine.md"),
                    "``\\ell^p`` distance" => joinpath("similarities", "lp_distance.md"),
                    "Jaccard similarity" => joinpath("similarities", "jaccard.md"),
                    "Inner product similarity" => joinpath("similarities", "inner_prod.md")],
                "Function-space hashing" => "function_hashing.md",
                "Performance tips" => "performance.md",
                "API reference" => "full_api.md",
                "FAQ" => "faq.md"
               ]
)

deploydocs(
    repo = "github.com/kernelmethod/LSHFunctions.jl.git"
)
