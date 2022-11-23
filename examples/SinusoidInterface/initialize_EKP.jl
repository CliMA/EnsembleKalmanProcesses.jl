include("shared.jl") # packages

function main()

    output_dir = joinpath(@__DIR__, ARGS[1])
    data_path = joinpath(output_dir, ARGS[2])
    eki_path = joinpath(output_dir, ARGS[3])
    toml_path = joinpath(@__DIR__, ARGS[4])
    N_ensemble = parse(Int64, ARGS[5])
    rng_seed = parse(Int64, ARGS[6])
    rng_ekp = Random.MersenneTwister(rng_seed)

    # We construct the prior from file
    param_dict = TOML.parsefile(toml_path)
    names = ["amplitude", "vert_shift"]
    prior_vec = [get_parameter_distribution(param_dict, n) for n in names]
    prior = combine_distributions(prior_vec)

    # we load the data, noise, from file
    @load data_path y Γ

    # initialize ensemble Kalman inversion
    initial_ensemble = EKP.construct_initial_ensemble(rng_ekp, prior, N_ensemble)
    eki = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); rng = rng_ekp)

    # save the parameter ensemble and EKP
    save_parameter_ensemble(
        get_u_final(eki), # constraints applied when saving
        prior,
        param_dict,
        output_dir,
        "parameters.toml",
        0, # We consider the initial ensemble to be the 0th iteration
    )

    #save new state
    @save eki_path eki param_dict prior


end

main()
