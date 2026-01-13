
### Can go in Lorenz96.jl


struct ConstantEMC{FT <: Real} <: EnsembleMemberConfig
    val::FT
end
build_forcing(val, args...) where {FT <: Real} = ConstantEMC(val[1])


struct VectorEMC{VV <: AbstractVector} <: EnsembleMemberConfig
    val::VV
end
build_forcing(val::FT, args...) where {FT <: Real} = VectorEMC(val)


struct FluxEMC <: EnsembleMemberConfig
    model::Flux.Chain
end
function build_forcing(model, params)
    _, reconstructor = Flux.destructure(model)
    return FluxEMC(reconstructor(params))
end


# Constant-global
forcing(params::ConstantEMC, x, i) = params.val

# Constant-vector
forcing(params::VectorEMC, x, i) = params.val[i]

# Flux
forcing(params::FluxEMC, x, i) = Float64(params.model([Float32(i)])[1])



### Can go in script.jl
# Will also need:
# Flux, BSON

## Define forcing:
cases = ["const-force", "vec-force", "flux-force"]
case = cases[1]

if case == "const-force"
    prior = constrained_gaussian("φ", 10.0, 4.0, 0, Inf)
    forcing = ConstantEMC(0.0)
    forcing_structure = nothing
elseif case == "vec-force"
    prior = ParameterDistribution(..)
    forcing = VectorEMC([0.0])
    forcing_structure = nothing
elseif case == "flux-force"
    from_file = true
    # from_file
    if from_file
        filename = "filename"
        forcing_structure = BSON.@load "$(filename).bson" model
        prior_mean, prior_cov = BSON.@load "$(filename).bson" prior_mean, prior_cov
        prior = ParameterDistribution(..)
    else
        input_dim = 1
        forcing_structure = Chain(Dense(input_dim => 20, tanh), Dense(20 => 1))
        prior = constrained_gaussian("params", 0, 1, -Inf, Inf, repeat = input_dim)
    end

    forcing = FluxEMC(model)
end



# ... later on.

# inside a loop over G_ens..
G_ens = zeros(..)
for j in 1:Ne
    forcing = build_forcing(params_i[j, :], model_structure)
    G_ens[j, :] = lorenz_forward(
        forcing,
        (x0 .+ ic_cov_sqrt * rand(rng, Normal(0.0, 1.0), nx, Ne))[:, j],
        lorenz_config_settings,
        observation_config,
    )
end
