#========================
Doctests
========================#

using Documenter, LSHFunctions, Test

@testset "LSH doctests" begin
    doctest(LSHFunctions; manual = false)
end
