using Test
using Distributions
using Random
using LinearAlgebra
using EnsembleKalmanProcesses.DataContainers

@testset "DataContainers" begin
    rng = Random.MersenneTwister(2021)

    parameter_samples = rand(rng, MvNormal(zeros(2), 0.1 * I), 10) #10 samples of 2D params
    data_samples = rand(rng, MvNormal(zeros(12), 2 * I), 10) #10 samples of 12D data
    data_samples_short = data_samples[:, 1:(end - 1)]

    new_parameter_samples = rand(rng, MvNormal(zeros(2), 0.1 * I), 10) #10 samples of 4D params
    new_data_samples = rand(rng, MvNormal(zeros(12), 2 * I), 10) #10 samples of 12D data
    new_data_samples_short = new_data_samples[:, 1:(end - 1)]

    #test DataContainer
    parameter_samples_T = permutedims(parameter_samples, (2, 1))
    idata = DataContainer(parameter_samples)
    @test idata == DataContainer(parameter_samples) # test ==
    idata_T = DataContainer(parameter_samples_T, data_are_columns = false)

    @test get_data(idata) == parameter_samples
    @test get_data(idata_T) == parameter_samples
    @test size(idata) == size(parameter_samples)

    #test Paired Container
    idata = DataContainer(parameter_samples)
    odata = DataContainer(data_samples)
    odata_short = DataContainer(data_samples_short)

    @test_throws DimensionMismatch PairedDataContainer(parameter_samples, data_samples_short)
    iopairs = PairedDataContainer(parameter_samples, data_samples)
    @test iopairs == PairedDataContainer(parameter_samples, data_samples) # test ==

    @test_throws DimensionMismatch PairedDataContainer(idata, odata_short)
    iopairs2 = PairedDataContainer(idata, odata)

    @test get_data(iopairs) == get_data(iopairs2)
    @test get_inputs(iopairs) == parameter_samples
    @test get_outputs(iopairs) == data_samples

    # test deep-copying
    parameter_samples[1, 1] += 1
    @test !isequal(get_data(idata), parameter_samples)
    @test !isequal(get_data(idata_T), parameter_samples)
    retrieved_samples = get_data(idata)
    retrieved_samples[1, 1] += 1
    @test !isequal(retrieved_samples, get_data(idata))
    retrieved_samples_T = get_data(idata_T)
    retrieved_samples_T[1, 1] += 1
    @test !isequal(retrieved_samples_T, get_data(idata_T))

    # test build from vectors and different types
    x = [1, 23, 4, 5, 6]
    xf = [1.4, 43.0, 23.0, 5.0, 9.0]

    @test_logs (:warn,) DataContainer(x) # vector is ambiguous treat as 1 x n
    dx = DataContainer(x)
    @test get_data(dx) == reshape(x, 1, :)
    dxf = DataContainer(xf)

    @test_logs (:warn,) PairedDataContainer(dx, dxf) # types clash, treat as promoted type
    pd1 = PairedDataContainer(dx, dxf)
    pd2 = PairedDataContainer(x, xf)
    
    @test pd1 == pd2
    @test eltype(get_inputs(pd1)) == promote_type(eltype(x), eltype(xf))
    @test eltype(get_outputs(pd1)) == promote_type(eltype(x), eltype(xf))

end
