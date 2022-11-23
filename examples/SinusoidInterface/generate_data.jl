include("observe_sinusoid.jl")

using LinearAlgebra, Distributions, JLD2

function main()

    # Paths
    output_dir = joinpath(@__DIR__, ARGS[1])
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    data_path = joinpath(output_dir, ARGS[2])
    rng_seed = parse(Int64, ARGS[3])
    rng_model = Random.MersenneTwister(rng_seed)

    # the true parameters
    theta_true = Dict("amplitude" => 1.0, "vert_shift" => 7.0)


    #randomness

    # create noise
    Γ = 0.1 * I
    dim_output = length(parameter_to_data_map(theta_true)) #just to get size here
    noise_dist = MvNormal(zeros(dim_output), Γ)

    # evaluate map with noise to create data
    y = parameter_to_data_map(theta_true, rng = rng_model) .+ rand(rng_model, noise_dist)

    # save
    @save data_path y Γ rng_model

end

main()
