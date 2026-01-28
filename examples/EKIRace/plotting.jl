# Plotting script for plotting race results: 
using StatsPlots
using Plots
using JLD2
using Statistics

data_filename = "output/l63_output_2026-01-27.jld2" # add filename here

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
N_ens_size = configuration["N_ens_sizes"]

ens_size_per_method = []
labels = String[]
for method in method_names
    if method == "Inversion(prior)"
        push!(labels, "TEKI")
        push!(ens_size_per_method, N_ens_size[ens_size_index])
    elseif method == "TransformInversion(prior)"
        push!(labels, "ETKI")
        push!(ens_size_per_method, N_ens_size[ens_size_index])
    elseif method == "GaussNewtonInversion(prior)"
        push!(labels, "GNKI")
        push!(ens_size_per_method, N_ens_size[ens_size_index])
    elseif method == "Unscented(prior; impose_prior=true)"
        push!(labels, "UKI")
        push!(ens_size_per_method, 1 +2*(size(final_parameters, 4)))
    end
end

avg_evaluations = mean(conv_alg_iters[:, ens_size_index, :], dims = 3)
target_rmse = configuration["target_rmse"]
typical_ens_size = N_ens_size[ens_size_index]

fow_run_plot = bar(
    labels,
    avg_evaluations,
    color = [:lightgreen, :deepskyblue3, :palevioletred1, :mediumpurple1],
    xlabel = "Method",
    ylabel = "Number of model evaluations",
    title  = "Target RMSE = $target_rmse and ensemble size = $typical_ens_size",
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
    color = [:lightgreen, :deepskyblue3, :palevioletred1, :mediumpurple1],
    xlabel = "Method",
    ylabel = "Number of iterations",
    title  = "Target RMSE = $target_rmse and ensemble size = $typical_ens_size",
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
