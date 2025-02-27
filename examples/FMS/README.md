

This example introduces you to how to use EnsembleKalmanProcesses.jl to calibrate your model and quantify uncertainties in a massively parallel environment:
  - Each EKP iteration requires evaluating the PDE solver N_ens times,	these evaluations can be embarrasingly paralleled.
  - Each PDE solver evaluation can be paralleled.


Here we have two examples
    - linear inverse problem
    - idealized GCM (https://github.com/szy21/fms_GCMForcing)


To run or Add new examples
    1) create your own `helper_funcs_XXXX.jl` file, which include 
       - `params_prefix` String:  parameter estimations for each iteration are saved in `params_prefix*string(iteration_)*".jld"`.

       - `input_prefix` String: EKP writes the input file for each PDE evalution in `input_prefix*string(i)*"/"*"input_file"`, here `i` is the ensemble particle id. This input file name can be adjusted in your function `write_solver_input_files` in `helper_funcs_XXXX.jl`.

       - `output_prefix` String: PDE solver writes the output for each PDE evalution in  `output_prefix*string(i)*"/"*"output_file.jld"`, here `i` is the ensemble particle id. This input file name can be adjusted in your function `write_solver_input_files` and `read_solver_output` in `helper_funcs_XXXX.jl`.

       - `input_file_template` String: template of your PDE solver input file.
       - `prior_mean` Array{Float64, 1}: prior mean.
       - `prior_cov` Array{Float64, 2}: prior covariance.
       - `N_params` Int64: number of parameters. 
       - `N_y` Int64: number of observations, it is N_obs + N_params, since the system is augmented to include prior information in EKP.
       - `param_bounds Vector{Vector}: lower and upper bounds for each parameter, when there is no bound, use `nothing`, like `[[u1_low, nothing], [nothing, nothing]]`.
`
       - `read_observation` Function: customer function to return observation mean and observation noise covariance.
       - `write_solver_input_files` Function: customer function to write input files for the PDE solver for each ensemble particle.
       - `read_solver_output` Function: customer function to read PDE prediction for the the corresponding ensemble `ens_index`.
       - `plot` Function : customer function to visualize the result 

    2) specify the number of particles `n` and the number of iterations `n_it` in `ekp_calibration.sbatch` 
    3) the PDE solver is called from `ekp_solver_run.sbatch`, you need to modify this function
    4) include `helper_funcs_XXXX.jl`
    5) submit the job `sbatch ekp_calibration.sbatch`
    6) the results will be saved in the `output` folder, you can use `plot` function in `help_funcs.jl` to visualize the results



