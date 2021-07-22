using JLD2
using PyPlot
using Statistics

# Parameters
outpath = pwd()
ekp_path = "ekp.jld2"
param_names = ["entrainment", "detrainment"]

# Load data
data = load(ekp_path)

# Mean
phi_m = mean(data["phi_params"], dims=3)[:,:,1]
# Variance
_uvar = var.(data["ekp_u"], dims=2)
n_iter = length(_uvar); n_param = length(_uvar[1])
uvar = zeros((n_iter, n_param))
for i in 1:n_iter uvar[i,:] = _uvar[i] end

# plot parameter evolution
fig, axs = subplots(nrows=n_param, sharex=true)
x = 0:n_iter-1
for (i, ax) in enumerate(axs)
    ax.plot(x, phi_m[:,i])
    ax.fill_between(x, 
        phi_m[:,i].-uvar[:,i], 
        phi_m[:,i].+uvar[:,i], 
        alpha=0.5,
    )
    ax.set_ylabel(param_names[i])
end

axs[1].set_xlim(0,n_iter-1)
axs[1].set_title("Parameter evolution")
axs[end].set_xlabel("iteration")
savefig(joinpath(outpath, "param_evol.png"))

# Error plot
x = 1:n_iter-1
err = data["ekp_err"]
fig, ax = subplots()
ax.plot(x, err)
ax.set_xlim(1,n_iter-1)
ax.set_ylabel("Error")
ax.set_xlabel("iteration")
ax.set_title("Error evolution")
savefig(joinpath(outpath, "error_evol.png"))

