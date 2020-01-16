using Pkg

Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()
Pkg.activate(); Pkg.instantiate()

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter, LSH

makedocs(
    sitename = "LSH.jl",
    format   = Documenter.HTML(),
    modules  = [LSH],
    pages    = ["Home" => "index.md"]
)

deploydocs(
    repo = "github.com/kernelmethod/LSH.jl.git"
)
