using Test
using Distributions
using Random

using EnsembleKalmanProcesses.DataStorage

@testset "DataStorage" begin
    seed=2021
    Random.seed!(seed)
        
    parameter_samples = rand(MvNormal(2,0.1),10) #10 samples of 4D params
    data_samples = rand(MvNormal(12,2),10) #10 samples of 12D data
    data_samples_short = data_samples[:,1:end-1]

    new_parameter_samples = rand(MvNormal(2,0.1),10) #10 samples of 4D params
    new_data_samples = rand(MvNormal(12,2),10) #10 samples of 12D data
    new_data_samples_short = new_data_samples[:,1:end-1]

    #test DataContainer
    parameter_samples_T = permutedims(parameter_samples, (2,1))
    idata = DataContainer(parameter_samples)
    idata_T = DataContainer(parameter_samples_T, data_are_columns=false)

    @test get_data(idata) == parameter_samples
    @test get_data(idata_T) == parameter_samples
    @test size(idata) == size(parameter_samples)

    #test Paired Container
    idata = DataContainer(parameter_samples)
    odata = DataContainer(data_samples)
    odata_short = DataContainer(data_samples_short)
    
    @test_throws DimensionMismatch PairedDataContainer(parameter_samples, data_samples_short)
    iopairs = PairedDataContainer(parameter_samples,data_samples)

    @test_throws DimensionMismatch PairedDataContainer(idata,odata_short)
    iopairs2 = PairedDataContainer(idata,odata)
    
    @test get_data(iopairs) == get_data(iopairs2)
    @test get_inputs(iopairs) == parameter_samples
    @test get_outputs(iopairs) == data_samples
    
end
