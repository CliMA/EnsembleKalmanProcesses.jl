using JLD2
using PyPlot
using Statistics

function make_ekp_plots(ekp_path::ST, param_names::Vector{ST}) where ST<:AbstractString
    n_param = length(param_names)

    # Load data
    data = load(joinpath(ekp_path, "ekp.jld2"))

    # Mean
    phi_m = mean(data["phi_params"], dims=3)[:,:,1]
    # Variance
    _ustd = std.(data["ekp_u"], dims=2)
    n_iter = length(_ustd); n_param = length(_ustd[1])
    ustd = zeros((n_iter, n_param))
    for i in 1:n_iter ustd[i,:] = _ustd[i] end

    # plot parameter evolution
    fig, axs = subplots(nrows=n_param, sharex=true, figsize=(15, 4*n_param))
    axs = n_param == 1 ? [axs] : axs
    x = 0:n_iter-1
    for (i, ax) in enumerate(axs)
        ax.plot(x, phi_m[:,i])
        ax.fill_between(x, 
            phi_m[:,i].-2ustd[:,i], 
            phi_m[:,i].+2ustd[:,i], 
            alpha=0.5,
        )
        ax.set_ylabel(param_names[i])
    end

    axs[1].set_xlim(0,n_iter-1)
    axs[1].set_title("Parameter evolution (mean ±2 std)")
    axs[end].set_xlabel("iteration")
    savefig(joinpath(ekp_path, "param_evol.png"))

    # Error plot
    x = 1:n_iter-1
    err = data["ekp_err"]
    fig, ax = subplots()
    ax.plot(x, err)
    ax.set_xlim(1,n_iter-1)
    ax.set_ylabel("Error")
    ax.set_xlabel("iteration")
    ax.set_title("Error evolution")
    savefig(joinpath(ekp_path, "error_evol.png"))
end
