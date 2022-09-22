using Test
using TOML
using Distributions
using Random
using LinearAlgebra

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.UQParameters
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions
const EKP = EnsembleKalmanProcesses

@testset "UQParameters" begin

    # Load parameters
    toml_path = joinpath(@__DIR__, "toml", "uq_test_parameters.toml")
    param_dict = TOML.parsefile(toml_path)

    # True `ParameterDistribution`s. This is what `get_parameter_distribution`
    # should return
    target_map = Dict(
        "uq_param_1" => ParameterDistribution(Parameterized(Normal(-100.0, 20.0)), no_constraint(), "uq_param_1"),
        "uq_param_2" => ParameterDistribution(Parameterized(Gamma(5.0, 2.0)), bounded_below(6.0), "uq_param_2"),
        "uq_param_3" => ParameterDistribution(
            Parameterized(MvNormal(4, 1.0)),
            [no_constraint(), bounded_below(-100.0), bounded_above(10.0), bounded(-42.0, 42.0)],
            "uq_param_3",
        ),
        "uq_param_4" => ParameterDistribution(
            Samples([5.0 3.2 4.8 3.6; -5.4 -4.7 -3.9 -4.5]),
            [bounded(0.0, 15.0), bounded_below(-10.0)],
            "uq_param_4",
        ),
        "uq_param_5" => ParameterDistribution(
            Samples([1.0 3.0; 5.0 7.0; 9.0 11.0; 13.0 15.0]),
            [no_constraint(), no_constraint(), bounded_below(-2.0), bounded_above(20.0)],
            "uq_param_5",
        ),
        "uq_param_6" => ParameterDistribution(
            VectorOfParameterized(repeat([Gamma(2.0, 3.0)], 3)),
            repeat([bounded_above(9.0)], 3),
            "uq_param_6",
        ),
        "uq_param_7" =>
            ParameterDistribution(Parameterized(MvNormal(3, 2.0)), repeat([no_constraint()], 3), "uq_param_7"),
        "uq_param_8" => ParameterDistribution(
            VectorOfParameterized([Gamma(2.0, 3.0), LogNormal(0.1, 0.1), Normal(0.0, 10.0)]),
            [no_constraint(), no_constraint(), bounded_below(-5.0)],
            "uq_param_8",
        ),
    )

    # Get all `ParameterDistribution`s. We also add dummy (key, value) pairs
    # to check if that information gets added correctly when saving the
    # parameters back to file and re-loading them
    uq_param_names = get_UQ_parameters(param_dict)
    descr = " will be learned using CES"

    # for test_throws:
    bad_toml_path = joinpath(@__DIR__, "toml", "bad_param.toml")
    bad_param_dict = TOML.parsefile(bad_toml_path)

    @test_throws ArgumentError get_parameter_distribution(bad_param_dict, "uq_param_baddist")
    @test_throws ArgumentError get_regularization(bad_param_dict, "uq_param_badL")

    for param_name in uq_param_names
        param_dict[param_name]["description"] = param_name * descr
        pd = get_parameter_distribution(param_dict, param_name)
        target_pd = target_map[param_name]

        # Check names
        @test get_name(pd) == get_name(target_pd)
        # Check distributions
        @test get_distribution(pd) == get_distribution(target_pd)
        # Check constraints
        constraints = get_all_constraints(pd)
        target_constraints = get_all_constraints(target_pd)
        @test constraints == target_constraints
    end

    # Check regularization flags
    @test get_regularization(param_dict, "uq_param_1") == ("L1", 1.5)
    @test get_regularization(param_dict, "uq_param_3") == ("L2", 1.1)
    @test get_regularization(param_dict, "uq_param_4") == (nothing, nothing)
    @test get_regularization(param_dict, ["uq_param_3", "uq_param_4"]) == [("L2", 1.1), (nothing, nothing)]

    # We can also get a `ParameterDistribution` representing
    # multiple parameters
    param_list = ["uq_param_2", "uq_param_4", "uq_param_5"]
    pd = get_parameter_distribution(param_dict, param_list)

    # Save the parameter dictionary and re-load it.
    param_dict_from_log = mktempdir(@__DIR__) do path
        logfile_path = joinpath(path, "log_file_test_uq.toml")
        write_log_file(param_dict, logfile_path)

        # Read in log file as new parameter file and rerun test.
        TOML.parsefile(logfile_path)
    end

    for param_name in uq_param_names
        pd = get_parameter_distribution(param_dict_from_log, param_name)
        @test get_distribution(pd) == get_distribution(target_map[param_name])
        @test param_dict_from_log[param_name]["description"] == param_name * descr
    end

    # ------
    # Test writing of EKP parameter ensembles
    # ------

    # Read parameters
    toml_path = joinpath(@__DIR__, "toml", "uq_test_parameters.toml")
    param_dict = TOML.parsefile(toml_path)

    # Extract the UQ parameters
    uq_param_names = get_UQ_parameters(param_dict)

    # Seed for pseudo-random number generator
    rng_seed = 42
    rng = Random.MersenneTwister(rng_seed)

    # Construct the parameter distribution
    pd = get_parameter_distribution(param_dict, uq_param_names)
    slices = batch(pd) # Will need this later to extract parameters

    # Create a ensemble Kalman process

    # First, generate symthetic observations y_obs by evaluating a 
    # (completely contrived) forward map G(u) (where u are the parameters) 
    # with the true parameter values u* (which we pretend to know for the
    # purpose of this example) and adding random observational noise η

    A3 = Array(reshape(rand!(rng, zeros(16)), 4, 4))
    A5 = Array(reshape(randn!(rng, zeros(16)), 4, 4))

    function G(u) # map from R^21 to R^4
        u_constr = transform_unconstrained_to_constrained(pd, u)
        value_of = Dict()
        for (i, param) in enumerate(get_name(pd))
            value_of[param] = u_constr[slices[i]]
        end
        A4 = reshape(
            [
                norm(value_of["uq_param_4"]) + norm(value_of["uq_param_6"]),
                norm(value_of["uq_param_7"]) + norm(value_of["uq_param_8"]),
                value_of["uq_param_2"][1],
                value_of["uq_param_1"][1],
            ],
            4,
            1,
        )
        y = (A3 * value_of["uq_param_3"] + A5 * value_of["uq_param_5"] + norm(value_of["uq_param_4"]) * A4)
        return dropdims(y, dims = 2)
    end

    # True parameter values (in constrained space)
    u1_star = 143.2
    u2_star = 8.3
    u3_star = [0.12, -0.05, -0.13, 0.05]
    u4_star = [12.0, 14.0]
    u5_star = [10.0, -1.0, 1.5, 10.0]
    u6_star = [3.0, 2.0, 8.0]
    u7_star = [5.0, 5.0, 10.0]
    u8_star = [-1.5, 2.3, 0.8]

    # Synthetic observation
    A4_star = reshape([norm(u4_star) + norm(u6_star), norm(u7_star) + norm(u8_star), u2_star, u1_star], 4, 1)

    y_star = A3 * u3_star + A5 * u5_star + norm(u4_star) * A4_star # G(u_star)
    Γy = 0.05 * I
    pdf_η = MvNormal(zeros(4), Γy)
    y_obs = dropdims(y_star, dims = 2) .+ rand(pdf_η)

    N_ens = 40 # number of ensemble members
    N_iter = 1 # number of iterations

    # Generate and save initial paramter ensemble
    initial_ensemble = construct_initial_ensemble(rng, pd, N_ens)
    mktempdir(@__DIR__) do save_path
        save_file = "test_parameters.toml"
        cov_init = cov(initial_ensemble, dims = 2)
        save_parameter_ensemble(
            initial_ensemble,
            pd,
            param_dict,
            save_path,
            save_file,
            0, # We consider the initial ensemble to be the 0th iteration
        )

        # Instantiate an ensemble Kalman process
        eki = EnsembleKalmanProcess(initial_ensemble, y_obs, Γy, Inversion(), rng = rng)

        # EKS iterations
        for i in 1:N_iter
            params_i = get_u_final(eki)
            G_n = [G(params_i[:, member_idx]) for member_idx in 1:N_ens]
            G_ens = hcat(G_n...)
            update_ensemble!(eki, G_ens)

            # Save updated parameter ensemble
            save_parameter_ensemble(get_u_final(eki), pd, param_dict, save_path, save_file, i)
        end

        # Check if all parameter files have been created (we expect there to be
        # one for each iteration and ensemble member)
        @test isdir(joinpath(save_path, "iteration_00"))
        @test isdir(joinpath(save_path, "iteration_01"))
        subdir_names = EKP.UQParameters.generate_subdir_names(N_ens)
        for i in 1:N_ens
            subdir_0 = joinpath(save_path, "iteration_00", subdir_names[i])
            subdir_1 = joinpath(save_path, "iteration_01", subdir_names[i])
            @test isdir(subdir_0)
            @test isfile(joinpath(subdir_0, save_file))
            @test isdir(subdir_1)
            @test isfile(joinpath(subdir_1, save_file))
        end
    end
end
