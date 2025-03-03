include("shared.jl") # packages

function main()

    output_dir = joinpath(@__DIR__, ARGS[1])
    eki_path = joinpath(output_dir, ARGS[2])

    # Parameters
    iteration = parse(Int64, ARGS[3])

    # load current state 
    @load eki_path eki param_dict prior
    N_ensemble = get_N_ens(eki)
    dim_output = size(get_obs(eki))[1]

    # load data from the ensemble
    G_ens = zeros(dim_output, N_ensemble)
    for member in 1:N_ensemble
        member_path = path_to_ensemble_member(output_dir, iteration, member)
        @load joinpath(member_path, "output.jld2") model_output
        G_ens[:, member] = model_output
    end

    # perform the update    
    EKP.update_ensemble!(eki, G_ens)

    # save the parameter ensemble and EKP
    save_parameter_ensemble(
        get_u_final(eki), # constraints applied when saving
        prior,
        param_dict,
        output_dir,
        "parameters",
        iteration + 1, #save for next iteration
    )

    #save new state
    @save eki_path eki param_dict prior

end

main()
