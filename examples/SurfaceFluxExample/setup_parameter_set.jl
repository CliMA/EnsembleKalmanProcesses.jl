# sets up parameters
# const sf_params = SF.Parameters.SurfaceFluxesParameters{
#     FT,
#     SF.UniversalFunctions.BusingerParams{FT},
#     TP.ThermodynamicsParameters{FT},
# }(
#     0.4f0,
#     SF.UniversalFunctions.BusingerParams{FT}(0.74f0, 4.7f0, 4.7f0, 2.5f0, 4.45f0),
#     TP.ThermodynamicsParameters{FT}(
#         273.16f0,
#         100000.0f0,
#         1859.0f0,
#         4181.0f0,
#         2100.0f0,
#         2.5008f6,
#         2.8344f6,
#         611.657f0,
#         273.16f0,
#         273.15f0,
#         150.0f0,
#         1000.0f0,
#         298.15f0,
#         6864.8f0,
#         10513.6f0,
#         0.2857143f0,
#         8.31446f0,
#         0.02897f0,
#         0.01801528f0,
#         290.0f0,
#         220.0f0,
#         9.80616f0,
#         233.0f0,
#         1.0f0,
#     ),
# )


import CLIMAParameters as CP
import SurfaceFluxes as SF
import SurfaceFluxes.UniversalFunctions as UF
import Thermodynamics as TD

function create_uf_parameters(toml_dict, ::UF.GryanikType)
    FT = CP.float_type(toml_dict)

    aliases = ["Pr_0_Gryanik", "a_m_Gryanik", "a_h_Gryanik", "b_m_Gryanik", "b_h_Gryanik", "ζ_a_Gryanik", "γ_Gryanik"]

    pairs = CP.get_parameter_values!(toml_dict, aliases, "UniversalFunctions")
    pairs = (; pairs...) # convert to NamedTuple

    pairs = (;
        Pr_0 = pairs.Pr_0_Gryanik,
        a_m = pairs.a_m_Gryanik,
        a_h = pairs.a_h_Gryanik,
        b_m = pairs.b_m_Gryanik,
        b_h = pairs.b_h_Gryanik,
        ζ_a = pairs.ζ_a_Gryanik,
        γ = pairs.γ_Gryanik,
    )
    return UF.GryanikParams{FT}(; pairs...)
end

function create_uf_parameters(toml_dict, ::UF.BusingerType)
    FT = CP.float_type(toml_dict)
    aliases = ["Pr_0_Businger", "a_m_Businger", "a_h_Businger", "ζ_a_Businger", "γ_Businger"]

    pairs = CP.get_parameter_values!(toml_dict, aliases, "UniversalFunctions")
    pairs = (; pairs...) # convert to NamedTuple

    pairs = (;
        Pr_0 = pairs.Pr_0_Businger,
        a_m = pairs.a_m_Businger,
        a_h = pairs.a_h_Businger,
        ζ_a = pairs.ζ_a_Businger,
        γ = pairs.γ_Businger,
    )
    return UF.BusingerParams{FT}(; pairs...)
end

function create_uf_parameters(toml_dict, ::UF.GrachevType)
    FT = CP.float_type(toml_dict)
    aliases = [
        "Pr_0_Grachev",
        "a_m_Grachev",
        "a_h_Grachev",
        "b_m_Grachev",
        "b_h_Grachev",
        "c_h_Grachev",
        "ζ_a_Grachev",
        "γ_Grachev",
    ]

    pairs = CP.get_parameter_values!(toml_dict, aliases, "UniversalFunctions")
    pairs = (; pairs...) # convert to NamedTuple

    pairs = (;
        Pr_0 = pairs.Pr_0_Grachev,
        a_m = pairs.a_m_Grachev,
        a_h = pairs.a_h_Grachev,
        b_m = pairs.b_m_Grachev,
        b_h = pairs.b_h_Grachev,
        c_h = pairs.c_h_Grachev,
        ζ_a = pairs.ζ_a_Grachev,
        γ = pairs.γ_Grachev,
    )
    return UF.GrachevParams{FT}(; pairs...)
end

function create_parameters(toml_dict, ufpt)
    FT = CP.float_type(toml_dict)

    ufp = create_uf_parameters(toml_dict, ufpt)
    AUFP = typeof(ufp)

    aliases = string.(fieldnames(TD.Parameters.ThermodynamicsParameters))
    pairs = CP.get_parameter_values!(toml_dict, aliases, "Thermodynamics")
    thermo_params = TD.Parameters.ThermodynamicsParameters{FT}(; pairs...)
    TP = typeof(thermo_params)

    aliases = ["von_karman_const"]
    pairs = CP.get_parameter_values!(toml_dict, aliases, "SurfaceFluxesParameters")
    override = return SFP.SurfaceFluxesParameters{FT, AUFP, TP}(; pairs..., ufp, thermo_params)
end

function override_climaatmos_defaults(defaults::NamedTuple, overrides::NamedTuple)
    intersect_keys = intersect(keys(defaults), keys(overrides))
    intersect_vals = getproperty.(Ref(overrides), intersect_keys)
    intersect_overrides = (; zip(intersect_keys, intersect_vals)...)
    return merge(defaults, intersect_overrides)
end
