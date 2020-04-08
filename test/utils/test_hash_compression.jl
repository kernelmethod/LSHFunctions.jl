using Test, LSHFunctions

#==================
Tests
==================#
@testset "HashCompressor tests" begin
    @testset "Can compress Vector{UInt8} hashes" begin
        compressor = HashCompressor(n_bytes=6)
        hashes = UInt8[0x01, 0x04, 0x02, 0x08, 0x06, 0x07, 0x08, 0x04]

        @test compressor(hashes) == UInt8[0xce, 0xd8, 0x24, 0x1c, 0xc0, 0x48]
    end

    @testset "Can compress Vector{Integer} hashes" begin
        compressor = HashCompressor(n_bytes=4)
        hashes = [-1, 8, -6, 3, -5, -9, 9, 0]

        @test compressor(hashes) == UInt8[0xb2, 0x7f, 0x8e, 0xb4]
    end

    @testset "Can compress BitArray{1} hashes" begin
        compressor = HashCompressor(n_bytes=5)
        hashes = BitArray([1, 1, 1, 0, 0, 1, 0, 0, 1, 0])

        @test compressor(hashes) == UInt8[0xa2, 0x99, 0xd7, 0x9f, 0x67]
    end

    @testset "Can salt hashes" begin
        salt = UInt8[0xcb, 0xe7, 0x12]
        compressor = HashCompressor(n_bytes=6, salt=salt)
        hashes = [-1, 8, -6, 3, -5, -9, 9, 0]

        @test compressor(hashes) == UInt8[0x9f, 0x5c, 0xf4, 0x3a, 0x29, 0x22]
    end
end
