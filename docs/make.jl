using EnsembleKalmanProcesses,
    Documenter,
    DocumenterCitations,
    Plots,  # so that Literate.jl does not capture precompilation output
    CairoMakie,
    Literate

prepend!(LOAD_PATH, [joinpath(@__DIR__, "..")])

# Gotta set this environment variable when using the GR run-time on CI machines.
# This happens as examples will use Plots.jl to make plots and movies.
# See: https://github.com/jheinen/GR.jl/issues/278
ENV["GKSwstype"] = "100"

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR = joinpath(@__DIR__, "src/literated")

examples_for_literation = [
    "Sinusoid/sinusoid_example.jl",
    "LossMinimization/loss_minimization.jl",
    "SparseLossMinimization/loss_minimization_sparse_eki.jl",
    "Darcy/darcy.jl",
]

if isempty(get(ENV, "CI", ""))
    # only needed when building docs locally; set automatically when built under CI
    # https://fredrikekre.github.io/Literate.jl/v2/outputformats/#Configuration
    extra_literate_config = Dict(
        "repo_root_path" => abspath(joinpath(@__DIR__, "..")),
        "repo_root_url" => "file://" * abspath(joinpath(@__DIR__, "..")),
    )
else
    extra_literate_config = Dict()
end

for example in examples_for_literation
    example_filepath = joinpath(EXAMPLES_DIR, example)
    Literate.markdown(
        example_filepath,
        OUTPUT_DIR;
        flavor = Literate.DocumenterFlavor(),
        config = extra_literate_config,
    )
end

#----------

api = [
    "ParameterDistributions" => "API/ParameterDistributions.md",
    "Observations" => "API/Observations.md",
    "DataContainers" => "API/DataContainers.md",
    "EnsembleKalmanProcess" => "API/EnsembleKalmanProcess.md",
    "Inversion" => "API/Inversion.md",
    "Unscented" => "API/Unscented.md",
    "Sampler" => "API/Sampler.md",
    "SparseInversion" => "API/SparseInversion.md",
    "TOML Interface" => "API/TOMLInterface.md",
    "Localizers" => "API/Localizers.md",
    "Visualize" => "API/Visualize.md",
]

# Navigation follows the Diátaxis framework (https://diataxis.fr/): tutorials (Examples),
# explanation (Methods), how-to guides (User guides), and reference (API, glossary)

examples = [
    "Simple example" => "literated/sinusoid_example.md",
    "Minimization Loss" => "literated/loss_minimization.md",
    "Darcy flow" => "literated/darcy.md",
    "Lorenz" => "examples/lorenz_example.md",
    "Cloudy" => "examples/Cloudy_example.md",
    "TOML interface" => "examples/sinusoid_example_toml.md",
    "Sparse Minimization Loss" => "literated/loss_minimization_sparse_eki.md",
]

methods = [
    "Ensemble Kalman Inversion" => "ensemble_kalman_inversion.md",
    "Gauss Newton Kalman Inversion" => "gauss_newton_kalman_inversion.md",
    "Ensemble Kalman Sampler" => "ensemble_kalman_sampler.md",
    "Unscented Kalman Inversion" => "unscented_kalman_inversion.md",
]

user_guides = [
    "List of default configurations" => "defaults.md",
    "Prior distributions" => "parameter_distributions.md",
    "Observations and Minibatching" => "observations.md",
    "Learning rate schedulers" => "learning_rate_scheduler.md",
    "Update Groups" => "update_groups.md",
    "Localization and SEC" => "localization.md",
    "Accelerators" => "accelerators.md",
    "Inflation" => "inflation.md",
    "Failure handling" => "failure_handling.md",
    "Parallelism and HPC" => "parallel_hpc.md",
    "Visualization" => "visualization.md",
]

reference = [
    "API" => api,
    "Internal data representation" => "internal_data_representation.md",
    "Glossary" => "glossary.md",
    "References" => "references.md",
]

pages = [
    "Home" => "index.md",
    "Installation instructions" => "installation_instructions.md",
    "Examples" => examples,
    "Methods" => methods,
    "User guides" => user_guides,
    "Troubleshooting" => "troubleshooting.md",
    "Reference" => reference,
    "Contributing" => "contributing.md",
]

#----------

format = Documenter.HTML(collapselevel = 1, prettyurls = !isempty(get(ENV, "CI", "")))

bib = CitationBibliography(joinpath(@__DIR__, "src", "bibliography.bib"), style = :authoryear)

makedocs(
    plugins = [bib],
    sitename = "EnsembleKalmanProcesses.jl",
    authors = "CliMA Contributors",
    format = format,
    pages = pages,
    modules = [EnsembleKalmanProcesses],
    doctest = true,
    clean = true,
    checkdocs = :exports,
)

if !isempty(get(ENV, "CI", ""))
    deploydocs(
        repo = "github.com/CliMA/EnsembleKalmanProcesses.jl.git",
        versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
        push_preview = true,
        devbranch = "main",
    )
end
