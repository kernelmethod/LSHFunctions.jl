#========================
Doctests
========================#

using Documenter, LSH, Test

@testset "LSH doctests" begin
    doctest(LSH; manual = false)
end
