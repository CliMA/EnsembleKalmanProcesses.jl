# Plotting script for plotting race results: 
using StatsPlots
using Plots
using JLD2
using Statistics

data_filename = "output/l96_output_vec-force_2026-01-30.jld2" # add filename here
@info "reading filename: $(data_filename)"

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

ens_size_index = 2 # add info statement here
@info "Index pickekd in ensemble sizes list: $ens_size_index"
N_ens_size = configuration["N_ens_sizes"]
@info "Plotting for ensemble size = $(N_ens_size[ens_size_index])"

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
    eval_err = std(conv_alg_iters[:, ens_size_index, :], dims = 2)
else 
    eval_err = zeros(length(labels))
end
@info "Error bars plotted for experiments with 3 or more trials (random seeeds)"
target_rmse = configuration["target_rmse"]

fow_run_plot = bar(
    labels,
    avg_evaluations,
    yerr = eval_err,
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
savefig(fow_run_plot, figure_save_directory * "fow_run_comparison.png")

iter_plot = bar(
    labels,
    avg_evaluations ./ ens_size_per_method,
    yerr = (eval_err ./ ens_size_per_method),
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
savefig(iter_plot, figure_save_directory * "number_of_iterations.png")



# p = plot(
#     configuration["N_ens_sizes"],
#     [mean(conv_alg_iters[1, :, :], dims = 2) mean(conv_alg_iters[2, :, :], dims = 2) mean(
#         conv_alg_iters[3, :, :],
#         dims = 2,
#     )],
#     label = ["EKI" "ETKI" "GNKI"],
#     color = [:lightgreen :deepskyblue3 :palevioletred1],
#     xlabel = "Ensemble size",
#     ylabel = "Number of forward runs",
#     title = "EKI Race",
#     show = true,
#     reuse = false,
# )
# plot!([81], [mean(conv_alg_iters[4, :, :])], label = "UKI", color = :mediumpurple1, marker = :square)
# readline()
# savefig(p, figure_save_directory * "fow_run_comparison.png")
