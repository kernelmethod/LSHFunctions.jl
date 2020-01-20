using Test, Random, LSHFunctions

@testset "Test intervals with LSHFunctions.@interval" begin
    Random.seed!(0)

    @testset "Construct an interval" begin
        # Create a new interval corresponding to [0,1)
        interval = LSHFunctions.@interval(0 ≤ x < 1)
        @test isa(interval, LSHFunctions.RealInterval)
        @test interval.lower == 0
        @test interval.upper == 1
        @test interval.closed_below == true
        @test interval.closed_above == false

        # Create another interval corresponding to (-Inf,Inf]
        interval = LSHFunctions.@interval(-Inf < x ≤ Inf)
        @test isa(interval, LSHFunctions.RealInterval)
        @test interval.lower == -Inf
        @test interval.upper == Inf
        @test interval.closed_below == false
        @test interval.closed_above == true

        # Construct an interval containing a single point
        interval = LSHFunctions.@interval(0 ≤ x ≤ 0)
        @test interval.lower == interval.upper == 0
        @test interval.closed_below == interval.closed_above == true

        # Construct an interval in reverse order
        @test LSHFunctions.@interval(1 > x ≥ 0) == LSHFunctions.@interval(0 ≤ x < 1)
        @test LSHFunctions.@interval(Inf > x ≥ -Inf) == LSHFunctions.@interval(-Inf ≤ x < Inf)

        # Construct interval using the <= symbols instead of ≤
        @test LSHFunctions.@interval(0 <= x <= 1) == LSHFunctions.@interval(0 ≤ x ≤ 1)
        @test LSHFunctions.@interval(1 >= x >= 0) == LSHFunctions.@interval(1 ≥ x ≥ 0)

        # Construct an interval using variables and expressions for the endpoints
        a, b = -1, 1
        @test LSHFunctions.@interval(a ≤ x ≤ b) == LSHFunctions.@interval(-1 ≤ x ≤ 1)
        @test LSHFunctions.@interval(2a ≤ x ≤ 2b) == LSHFunctions.@interval(-2 ≤ x ≤ 2)

        # Construct an interval that contains a different symbol in the middle
        @test LSHFunctions.@interval(0 ≤ y < 1) == LSHFunctions.@interval(0 ≤ x < 1)
        @test LSHFunctions.@interval(0 ≤ mynum < 1) == LSHFunctions.@interval(0 ≤ x < 1)
    end

    @testset "Incorrect interval construction raises errors" begin
        # Try to construct intervals using invalid expressions
        @test_throws(ErrorException, LSHFunctions.@interval(0))
        @test_throws(ErrorException, LSHFunctions.@interval(0 ≤ x))
        @test_throws(ErrorException, LSHFunctions.@interval(0 ≤ 1 ≤ 2))

        # Try to construct intervals using inequalities pointing in different directions
        @test_throws(ErrorException, LSHFunctions.@interval(0 ≤ x ≥ 1))
        @test_throws(ErrorException, LSHFunctions.@interval(0 > x < 1))

        # Try to construct intervals not on the real line
        @test_throws(MethodError, LSHFunctions.@interval(-1im ≤ x ≤ 1))
        @test_throws(MethodError, LSHFunctions.@interval(0 ≤ x ≤ 1im))
        @test_throws(MethodError, LSHFunctions.@interval(-1im ≤ x ≤ 1im))

        # Try to construct intervals using undefined endpoints
        @test_throws(UndefVarError, LSHFunctions.@interval(a ≤ x ≤ Inf))
        @test_throws(UndefVarError, LSHFunctions.@interval(-Inf ≤ x ≤ b))
    end

    @testset "Test interval methods" begin
        # isempty
        @test isempty(LSHFunctions.@interval(0 ≤ x < -1))
        @test isempty(LSHFunctions.@interval(0 ≤ x < 0))

        # width
        @test LSHFunctions.width(LSHFunctions.@interval(0 ≤ x < 1)) == 1
        @test LSHFunctions.width(LSHFunctions.@interval(0 ≤ x < Inf)) == Inf
        @test LSHFunctions.width(LSHFunctions.@interval(0 ≤ x < -1)) == 0
    end

    @testset "Test membership within an interval" begin
        # Test 1: interval [0,1)
        interval = LSHFunctions.@interval(0 ≤ x < 1)

        @test all(x ∈ interval for x in rand(32))
        @test 1 ∉ interval

        # Test 2: interval (-Inf, Inf]
        interval = LSHFunctions.@interval(-Inf < x ≤ Inf)

        @test all(log(x) ∈ interval for x in rand(32))
        @test Inf ∈ interval
        @test -Inf ∉ interval
    end

    @testset "Test interval intersection" begin
        @test LSHFunctions.@interval(0 < x ≤ 1) ∩ LSHFunctions.@interval(0 ≤ x ≤ 1) ==
              LSHFunctions.@interval(0 < x ≤ 1)
        @test LSHFunctions.@interval(0 < x ≤ 1) ∩ LSHFunctions.@interval(1/2 ≤ x < 1) ==
              LSHFunctions.@interval(1/2 ≤ x < 1)

        # Ensure that the intersect() function is equal to ∩
        @test let success = true
            for ii = 1:128
                a, c = rand(), rand()
                b, d = a + rand(), c + rand()
                interval_1 = LSHFunctions.@interval(a < x ≤ b)
                interval_2 = LSHFunctions.@interval(c ≤ x < d)
                success &= (LSHFunctions.intersect(interval_1, interval_2) == interval_1 ∩ interval_2)
            end
            success
        end

        # Intersect two disjoint intervals to create an empty interval
        @test isempty(LSHFunctions.@interval(0 < x < 1) ∩ LSHFunctions.@interval(1 < x < 2))
    end

    @testset "Test empty intervals" begin
        # Test for equality between empty intervals
        @test LSHFunctions.@interval(0 < x < 0) == LSHFunctions.@interval(1 < x < 1)
        @test LSHFunctions.@interval(0 < x < -1) == LSHFunctions.@interval(0 < x < 0)
        @test LSHFunctions.@interval(0 < x < 0) == LSHFunctions.@interval(0 < x < -1)
    end
end
