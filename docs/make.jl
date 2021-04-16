# reference in tree version of CalibrateEmulateSample
prepend!(LOAD_PATH, [joinpath(@__DIR__, "..")])

using EnsembleKalmanProcesses
using Documenter
# using DocumenterCitations

#----------

api = ["EnsembleKalmanProcessModule" => "API/EnsembleKalmanProcessModule.md",
       "ParameterDistribution" => "API/ParameterDistribution.md",
       "Observations" => "API/Observations.md",
       "DataStorage" => "API/DataStorage.md",
] 

examples = ["Template example" => "examples/template_example.md"]

pages = [
    "Home" => "index.md",
    "Installation instructions" => "installation_instructions.md",
    "Prior distributions" => "parameter_distributions.md",
    "Observations" => "observations.md",
    "Ensemble Kalman Inversion" => "ensemble_kalman_inversion.md",
    "Examples" => examples,
    "API" => api,
]

#----------

format = Documenter.HTML(
    collapselevel = 1,
    prettyurls = !isempty(get(ENV, "CI", ""))
)

makedocs(
  sitename = "EnsembleKalmanProcesses.jl",
  authors = "CliMA Contributors",
  format = format,
  pages = pages,
  modules = [EnsembleKalmanProcesses],
  doctest = false,
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
