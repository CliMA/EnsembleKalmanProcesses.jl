# Plotting script for plotting race results: 
using StatsPlots
using Plots
using JLD2
using Statistics

plot_id = "l63" # id in output filename
data_filename = "output/l63_output_2026-01-30.jld2" # add filename here
@info "reading filename: $(data_filename)"
@info "plotting id: $(plot_id)"

data = JLD2.load(data_filename)
configuration        = data["configuration"]
method_names         = data["method_names"]
conv_alg_iters       = data["conv_alg_iters"]
final_parameters     = data["final_parameters"]
final_model_output   = data["final_model_output"]

# Output figure save directory
homedir = pwd()
println(homedir)
figure_save_directory = homedir * "/output/"
data_save_directory = homedir * "/output/"
if ~isdir(figure_save_directory)
    mkdir(figure_save_directory)
end
if ~isdir(data_save_directory)
    mkdir(data_save_directory)
end

ens_size_index = 1
@info "Index picked in ensemble sizes list: $ens_size_index"
N_ens_size = configuration["N_ens_sizes"]
@info "Plotting for ensemble size = $(N_ens_size[ens_size_index]) of $(N_ens_size)"

ens_size_per_method = []
labels = String[]
for method in method_names
    if method[2] == "UKI"
        push!(labels, method[2] * " ($(1 +2*(size(final_parameters, 4))))")
        push!(ens_size_per_method, 1 +2*(size(final_parameters, 4)))
    else
        push!(labels, method[2] * " ($(N_ens_size[ens_size_index]))")
        push!(ens_size_per_method, N_ens_size[ens_size_index])
    end
end

avg_evaluations = mean(conv_alg_iters[:, ens_size_index, :], dims = 2)
if size(conv_alg_iters)[3] >= 3
    std_evaluations = std(conv_alg_iters[:, ens_size_index, :], dims = 2)
    eval_err_low = min.(avg_evaluations, std_evaluations)
    eval_err_high = std_evaluations
else 
    eval_err_low = zeros(length(labels))
    eval_err_high = zeros(length(labels))
end
@info "Error bars plotted for experiments with 3 or more trials (random seeeds)"
target_rmse = configuration["target_rmse"]

fow_run_plot = bar(
    labels,
    avg_evaluations,
    yerr = [eval_err_low'; eval_err_high']',
    color = [:lightgreen, :deepskyblue3, :palevioletred1, :mediumpurple1],
    xlabel = "Method (ensemble size)",
    ylabel = "Average number of model evaluations",
    title  = "Target RMSE = $target_rmse",
    bar_width = 0.7,
    alpha = 0.8,
    legend = false,
    show = true,
    reuse = false,
)
readline()
savefig(fow_run_plot, joinpath(figure_save_directory, "$(plot_id)_ens$(N_ens_size[ens_size_index])_" * "fow_run_comparison.png"))

iter_plot = bar(
    labels,
    avg_evaluations ./ ens_size_per_method,
    yerr = ([eval_err_low'; eval_err_high']' ./ ens_size_per_method),
    color = [:lightgreen, :deepskyblue3, :palevioletred1, :mediumpurple1],
    xlabel = "Method (ensemble size)",
    ylabel = "Average number of iterations",
    title  = "Target RMSE = $target_rmse",
    bar_width = 0.7,
    alpha = 0.8,
    legend = false,
    show = true,
    reuse = false,
)
readline()
savefig(iter_plot, joinpath(figure_save_directory, "$(plot_id)_ens$(N_ens_size[ens_size_index])_" * "number_of_iterations.png"))
