module GModel

export run_G
export run_G_ensemble
export lorenz_forward

include("GModel_common.jl")

function run_ensembles(settings, lorenz_params, nd, N_ens)
    g_ens = zeros(nd, N_ens)
    for i in 1:N_ens
        # run the model with the current parameters, i.e., map θ to G(θ)
        g_ens[:, i] = lorenz_forward(settings, lorenz_params[i])
    end
    return g_ens
end

end # module
