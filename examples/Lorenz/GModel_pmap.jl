@everywhere module GModel

using Distributed

export run_G
export run_G_ensemble
export lorenz_forward

include("GModel_common.jl")

function run_ensembles(settings, lorenz_params, nd, N_ens)
    g_ens = zeros(nd, N_ens)
    g_ens[:, :] = vcat(pmap(x -> lorenz_forward(settings, x), lorenz_params)...)
    return g_ens
end

end # module
