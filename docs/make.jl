# reference in tree version of CalibrateEmulateSample
prepend!(LOAD_PATH, [joinpath(@__DIR__, "..")])

using EnsembleKalmanProcesses,
    Documenter,
    Plots,  # so that Literate.jl does not capture precompilation output
    Literate

# Gotta set this environment variable when using the GR run-time on CI machines.
# This happens as examples will use Plots.jl to make plots and movies.
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = "100"

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR = joinpath(@__DIR__, "src/literated")

examples_for_literation = ["LossMinimization/loss_minimization.jl", "AerosolActivation/aerosol_activation.jl"]

for example in examples_for_literation
    example_filepath = joinpath(EXAMPLES_DIR, example)
    Literate.markdown(example_filepath, OUTPUT_DIR; flavor = Literate.DocumenterFlavor())
end

#----------

api = [
    "EnsembleKalmanProcessModule" => "API/EnsembleKalmanProcessModule.md",
    "ParameterDistribution" => "API/ParameterDistribution.md",
    "Observations" => "API/Observations.md",
    "DataStorage" => "API/DataStorage.md",
]

examples = [
    "Template" => "examples/template_example.md",
    "Cloudy" => "examples/Cloudy_example.md",
    "Lorenz" => "examples/lorenz_example.md",
    "Minimization Loss" => "literated/loss_minimization.md",
    "Aerosol activation" => "literated/aerosol_activation.md",
    "HPC interfacing example: ClimateMachine" => "examples/ClimateMachine_example.md",
]

pages = [
    "Home" => "index.md",
    "Installation instructions" => "installation_instructions.md",
    "Prior distributions" => "parameter_distributions.md",
    "Observations" => "observations.md",
    "Ensemble Kalman Inversion" => "ensemble_kalman_inversion.md",
    "Ensemble Kalman Sampler" => "ensemble_kalman_sampler.md",
    "Unscented Kalman Inversion" => "unscented_kalman_inversion.md",
    "Examples" => examples,
    "API" => api,
    "Glossary" => "glossary.md",
]

#----------

format = Documenter.HTML(collapselevel = 1, prettyurls = !isempty(get(ENV, "CI", "")))

makedocs(
    sitename = "EnsembleKalmanProcesses.jl",
    authors = "CliMA Contributors",
    format = format,
    pages = pages,
    modules = [EnsembleKalmanProcesses],
    doctest = true,
    strict = true,
    clean = true,
    checkdocs = :none,
)

if !isempty(get(ENV, "CI", ""))
    deploydocs(
        repo = "github.com/CliMA/EnsembleKalmanProcesses.jl.git",
        versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
        push_preview = true,
        devbranch = "main",
    )
end
