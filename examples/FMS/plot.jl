using JLD
using PyPlot
# Import EnsembleKalmanProcesses modules
using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions

include(joinpath(@__DIR__, "helper_funcs.jl"))


N_iterations = 5



constraint_u_mean_all, constraint_uu_cov_all = read_all_params(N_iterations)
constraint_u_mean_all_arr, constraint_uu_cov_all_arr = zeros(N_params, N_iterations+1), zeros(N_params, N_iterations+1)

for iteration_ = 0: N_iterations
    
    constraint_u_mean_all_arr[:, iteration_+1] = constraint_u_mean_all[:, iteration_+1]
    constraint_uu_cov_all_arr[:, iteration_+1] = diag(constraint_uu_cov_all[:,:, iteration_+1])
end



ites = Array(0:N_iterations)

figure(figsize = (7.5, 4.8))

for i = 1: N_params

    errorbar(ites, constraint_u_mean_all_arr[i,:], yerr=3.0*constraint_uu_cov_all_arr[i,:], fmt="--o",fillstyle="none", label=L"$(u_names[i])")

    semilogy(ites, fill(u_refs[i], N_iterations+1), "--", color="gray")

end    
  

xlabel("Iterations")
legend(bbox_to_anchor=(0.95, 0.8))
grid("on")
tight_layout()
savefig("uki_parameters.pdf")
close("all")




uki_errors = zeros(Float64, 2, N_iterations+1)
for iteration_ = 1:N_iterations+1
    uki_errors[1, iteration_] = norm(constraint_u_mean_all[:,iteration_] - u_mean_post)
    uki_errors[2, iteration_] = norm(constraint_uu_cov_all[:,:,iteration_] - uu_cov_post)
end
    
figure(figsize = (7.5, 4.8))


semilogy(ites, uki_errors[1, :], "--o", fillstyle="none", label="mean")
semilogy(ites, uki_errors[2, :], "--o", fillstyle="none", label="cov")


xlabel("Iterations")
legend(bbox_to_anchor=(0.95, 0.8))
grid("on")
tight_layout()
savefig("uki_convergence.pdf")
close("all")

