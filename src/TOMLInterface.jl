module TOMLInterface

# Imports
using TOML
using Distributions
using DocStringExtensions
using EnsembleKalmanProcesses.ParameterDistributions

# Exports
export path_to_ensemble_member
export get_parameter_distribution
export get_parameter_values
export save_parameter_ensemble
export get_admissible_parameters
export get_regularization
export write_log_file
export save_parameter_samples

"""
$(TYPEDSIGNATURES)

Return parameter values from `param_dict`, indexed by the names in `names`.

# Arguments
- `param_dict`: nested dictionary mapping parameter names to sub-dictionaries that each contain a `"value"` key.
- `names`: iterable of parameter names to retrieve.
- `return_type`: `"dict"` (default) returns a `Dict{name => value}`; `"array"` returns a plain vector.
"""
function get_parameter_values(param_dict::Dict, names; return_type = "dict")
    if return_type == "dict"
        return Dict(n => param_dict[n]["value"] for n in names)
    elseif return_type == "array"
        return [param_dict[n]["value"] for n in names]
    else
        throw(ArgumentError("""
Unknown return_type for get_parameter_values.

Expected:
    "dict" or "array"

Got:
    return_type = $(repr(return_type))
"""))
    end
end

"""
$(TYPEDSIGNATURES)

Construct and return a `ParameterDistribution` for the parameter `name` from `param_dict`.

# Arguments
- `param_dict`: nested dictionary mapping parameter names to sub-dictionaries that each contain `"prior"` and `"constraint"` keys.
- `name`: name of the parameter to build a distribution for.
"""
function get_parameter_distribution(param_dict::Dict, name::AbstractString)

    # Constructing a parameter distribution requires a prior distribution,
    # a constraint, and a name.
    prior = construct_prior(param_dict[name])
    # If constrained_gaussian, then prior is already a ParameterDistribution
    if prior isa ParameterDistribution
        return prior
    end
    constraint = construct_constraint(param_dict[name])

    return ParameterDistribution(prior, constraint, name)
end


function get_parameter_distribution(param_dict::Dict, names::AbstractVector{String})

    param_dist_arr = map(names) do name

        get_parameter_distribution(param_dict, name)

    end

    return combine_distributions(param_dist_arr)

end


"""
$(TYPEDSIGNATURES)

Construct a `Constraint` from the `"constraint"` entry in `param_info`.

# Arguments
- `param_info`: dictionary containing at least a `"constraint"` key whose value is a constraint expression string as parsed from a TOML file.
"""
function construct_constraint(param_info::Dict)

    haskey(param_info, "constraint") || throw(ArgumentError("""
Parameter info dict is missing the required "constraint" key.

Got keys:
    $(collect(keys(param_info)))

Suggestion:
    Ensure the TOML entry for this parameter includes a `constraint = ...` field.
"""))
    c = Meta.parse(param_info["constraint"])

    if c.args[1] == Symbol("repeat")
        # Constraints are given as a `repeat` expression defining a vector of
        # constraints of the same kind
        constr = collect_from_expr(c.args[2], "c", repeat = true)
        n_constr = c.args[3] # number of repeated constraints
        return repeat([constr], n_constr)

    else
        return collect_from_expr(c, "c")
    end
end


"""
$(TYPEDSIGNATURES)

Construct a prior distribution from the `"prior"` entry in `param_info`.

Returns a distribution of type `Parameterized`, `Samples`, or `VectorOfParameterized`.

# Arguments
- `param_info`: dictionary containing at least a `"prior"` key whose value is a distribution expression string as parsed from a TOML file.
"""
function construct_prior(param_info::Dict)

    haskey(param_info, "prior") || throw(ArgumentError("""
Parameter info dict is missing the required "prior" key.

Got keys:
    $(collect(keys(param_info)))

Suggestion:
    Ensure the TOML entry for this parameter includes a `prior = ...` field.
"""))
    d = Meta.parse(param_info["prior"])

    if d.args[1] == Symbol("VectorOfParameterized")
        # There are multiple distributions described by a
        # `VectorOfParameterized`
        return get_vector_of_parameterized(d)

    else
        # Single distribution
        return get_distribution_from_expr(d)
    end
end


"""
$(TYPEDSIGNATURES)

Parse a `VectorOfParameterized` distribution from expression `d`.
"""
function get_vector_of_parameterized(d::Expr)

    d.args[1] == Symbol("VectorOfParameterized") || error(
        "Internal error: get_vector_of_parameterized called with non-VectorOfParameterized expression (got $(d.args[1]))",
    )

    if d.args[2].args[1] == Symbol("repeat")
        # Distributions are given as a `repeat` expression defining a vector of
        # distributions of the same kind
        dist = collect_from_expr(d.args[2].args[2], "d", repeat = true)
        n_dist = d.args[2].args[3] # number of repeated distributions
        return VectorOfParameterized(repeat([dist], n_dist))

    else
        # Distributions are given as an array of distributions listing each
        # individual distribution explicitly
        dist_arr = collect_from_expr(d.args[2], "d")
        return VectorOfParameterized(dist_arr)
    end

end


"""
$(TYPEDSIGNATURES)

Collect distributions or constraints from expression `e`.

# Arguments
- `e`: expression containing distribution or constraint information.
- `eltype`: `"d"` for distributions, `"c"` for constraints.
- `repeat`: `true` if the expression is a `repeat(...)` form.
"""
function collect_from_expr(e::Expr, eltype::AbstractString; repeat::Bool = false)

    if e.head == Symbol("vect")
        # There are multiple distributions / constraints 
        n_elem = length(e.args) # number of elements
        arr = (eltype == "d") ? Array{Distribution}(undef, n_elem) : Array{ConstraintType}(undef, n_elem)

        for i in 1:n_elem
            elem = e.args[i]
            arr[i] = getfield(ParameterDistributions, elem.args[1])(elem.args[2:end]...)
        end

        return repeat ? arr[1] : arr

    else
        # There is a single distribution / constraint
        return getfield(ParameterDistributions, e.args[1])(e.args[2:end]...)
    end

end


"""
$(TYPEDSIGNATURES)

Parse a `Parameterized`, `Samples`, or `constrained_gaussian` distribution from expression `d`.
"""
function get_distribution_from_expr(d::Expr)

    dist_type_symb = d.args[1]

    if dist_type_symb == Symbol("Parameterized")
        dist = getfield(Distributions, d.args[2].args[1])
        dist_args = d.args[2].args[2:end]
        dist_type = getfield(ParameterDistributions, dist_type_symb)

        return dist_type(dist(dist_args...))

    elseif dist_type_symb == Symbol("Samples")
        dist_args = construct_2d_array(d.args[2])
        dist_type = getfield(ParameterDistributions, dist_type_symb)

        return dist_type(dist_args)

    elseif dist_type_symb == Symbol("constrained_gaussian")

        function parse_kwargs(args)
            kwargs = []
            for arg in args
                # Only parse repeats kwarg for now
                arg.args[1] != :repeats && throw(ArgumentError("""
                       Unsupported keyword argument in constrained_gaussian TOML entry.

                       Expected:
                           repeats (the only keyword argument supported by the TOML parser)

                       Got:
                           $(arg.args[1])

                       Suggestion:
                           Remove unsupported keyword arguments from the TOML entry, or use the Julia API directly.
                       """))
                push!(kwargs, arg.args[1] => parse(Int64, string(arg.args[2])))
            end
            return kwargs
        end

        kwargs = []
        index = 2
        # Non-positional kwargs are the second argument
        if d.args[2] isa Expr && d.args[2].args[1].head == :kw
            kwargs = parse_kwargs(d.args[2].args)
            index += 1
            # Positional kwargs
        elseif length(d.args) > 6 && d.args[7].head == :kw
            kwargs = parse_kwargs([d.args[7]])
        end

        name, dist_mean, dist_std, lb, ub = d.args[index:(index + 4)]
        name = string(name)
        lower_bound, upper_bound = parse.(Float64, string.((lb, ub)))
        return constrained_gaussian(name, dist_mean, dist_std, lower_bound, upper_bound; kwargs...)

    else
        throw(ArgumentError("Unknown distribution type from symbol: $(dist_type_symb)"))
    end
end


"""
$(TYPEDSIGNATURES)

Reconstruct a 2D `Float64` sample matrix from a `vcat` expression `arr`.
"""
function construct_2d_array(arr::Expr)

    arr.head == Symbol("vcat") || throw(ArgumentError("""
Samples array expression must represent a 2D matrix (a `vcat` expression).

Got:
    arr.head = $(arr.head)

Suggestion:
    Ensure the TOML `samples` field is formatted as a 2D matrix, not a 1D vector.
"""))
    n_rows = length(arr.args)
    arr_of_rows = [arr.args[i].args for i in 1:n_rows]

    return Float64.(vcat(arr_of_rows'...))
end


"""
$(TYPEDSIGNATURES)

Sample `num_samples` draws from `distribution` and save each as a separate TOML file under `save_path`.

# Arguments
- `distribution`: `ParameterDistribution` to sample from.
- `default_param_data`: `Dict` of default parameters merged with the sampled values before writing.
- `num_samples`: number of samples to draw and save.
- `save_path`: directory under which per-sample TOML files are written.
- `save_file`: filename to use for each TOML file (default `"parameters.toml"`).
- `rng`: random-number generator (default `MersenneTwister(1234)`).
- `pad_zeros`: number of digits used for zero-padding ensemble-member indices.
"""
function save_parameter_samples(
    distribution::ParameterDistribution,
    default_param_data,
    num_samples,
    save_path;
    save_file = "parameters.toml",
    rng = Random.MersenneTwister(1234),
    pad_zeros = 3,
)

    save_parameter_ensemble(
        sample(rng, distribution, num_samples),
        distribution,
        default_param_data::Dict,
        save_path,
        save_file;
        pad_zeros,
        apply_constraints = true,
    )
end

"""
$(TYPEDSIGNATURES)

Save the parameter ensemble in `param_array` to TOML files organised by iteration and ensemble member.

Creates `<save_path>/iteration_<iter>/member_<j>/` for each member `j` and writes a TOML file
named `save_file` into each subdirectory. Parameter values are transformed from the unconstrained
to the constrained space when `apply_constraints` is `true`.

# Arguments
- `param_array`: `N_param × N_ens` matrix of parameter values in the unconstrained space.
- `param_distribution`: `ParameterDistribution` describing the parameters in `param_array`.
- `default_param_data`: `Dict` of default parameters merged with ensemble values before writing.
- `save_path`: root directory under which iteration and member subdirectories are created.
- `save_file`: filename for the TOML file written in each member directory.
- `iteration`: current iteration index, used to name the `iteration_<iter>` subdirectory.
- `pad_zeros`: number of digits used for zero-padding directory indices.
- `apply_constraints`: apply constraints in `param_distribution` before saving (default `true`).
"""
function save_parameter_ensemble(
    param_array::Array{FT, 2},
    param_distribution::ParameterDistribution,
    default_param_data::Dict,
    save_path::AbstractString,
    save_file::AbstractString,
    iteration::Int;
    pad_zeros = 3,
    apply_constraints = true,
) where {FT}

    save_dir = joinpath(save_path, join(["iteration", lpad(iteration, pad_zeros, "0")], "_"))
    return save_parameter_ensemble(
        param_array,
        param_distribution,
        default_param_data,
        save_dir,
        save_file;
        pad_zeros = pad_zeros,
        apply_constraints = apply_constraints,
    )
end

function save_parameter_ensemble(
    param_array::Array{FT, 2},
    param_distribution::ParameterDistribution,
    default_param_data::Dict,
    save_path::AbstractString,
    save_file::AbstractString;
    pad_zeros = 3,
    apply_constraints = true,
) where {FT}

    # The parameter values are currently in the unconstrained space
    # where the ensemble Kalman algorithm takes place
    if apply_constraints
        save_array = transform_unconstrained_to_constrained(param_distribution, param_array)
    else
        save_array = param_array
    end

    # The number of rows in param_array represent the sum of all parameter
    # dimensions. We need to determine the slices of rows that belong to
    # each parameter. E.g., an array with 6 rows could be sliced into
    # one 1-dim parameter (first row), one 3-dim parameter (rows 2 to 4),
    # and a 2-dim parameter (rows 5 to 6)
    param_slices = batch(param_distribution)
    param_names = get_name(param_distribution)

    N_ens = size(save_array)[2]

    # Create directory where files will be stored if it doesn't exist yet
    mkpath(save_path)

    # Each ensemble member gets its own subdirectory
    subdir_names = generate_subdir_names(N_ens, mode = "all", pad_zeros = pad_zeros)

    # All parameter toml files (one for each ensemble member) have the same name
    toml_file = endswith(save_file, ".toml") ? save_file : save_file * ".toml"

    for i in 1:N_ens
        mkpath(joinpath(save_path, subdir_names[i]))
        # Override the value (or add a value, if no value exists yet)
        # of the parameter in the original parameter dict with the
        # corresponding value in param_array
        param_dict = deepcopy(default_param_data)

        param_dict_updated = assign_values!(i, save_array, param_distribution, param_slices, param_dict, param_names)

        open(joinpath(save_path, subdir_names[i], toml_file), "w") do io
            TOML.print(io, param_dict_updated)
        end
    end
end

"""
$(TYPEDSIGNATURES)

Return the file-system path to ensemble member `member` at the given `iteration`.

The returned path has the form `<base_path>/iteration_<iter>/member_<j>/`,
with both indices zero-padded to `pad_zeros` digits.

# Arguments
- `base_path`: root directory where EKP outputs are stored.
- `iteration`: iteration index of the ensemble update.
- `member`: one-based ensemble member index.
- `pad_zeros`: number of digits used for zero-padding (default `3`).
"""
function path_to_ensemble_member(base_path::AbstractString, iteration::Int, member::Int; pad_zeros = 3)

    # Get the directory of the iteration
    base_dir = joinpath(base_path, join(["iteration", lpad(iteration, pad_zeros, "0")], "_"))
    return path_to_ensemble_member(base_dir, member, pad_zeros = pad_zeros)
end

function path_to_ensemble_member(base_path::AbstractString, member::Int; pad_zeros = 3)
    # get the directory of the member
    subdir_name = generate_subdir_names(member, mode = "only", pad_zeros = pad_zeros)
    return joinpath(base_path, subdir_name)
end

"""
$(TYPEDSIGNATURES)

Update `param_dict` in-place with the values of ensemble member `member` from `param_array`.

# Arguments
- `member`: column index of the ensemble member in `param_array`.
- `param_array`: `N_par × N_ens` array of parameter values.
- `param_distribution`: `ParameterDistribution` describing the parameters in `param_array`.
- `param_slices`: contiguous index ranges splitting `param_array` rows by parameter dimension.
- `param_dict`: parameter dictionary to update in-place.
- `names`: parameter names corresponding to each slice.
"""
function assign_values!(
    member::Int,
    param_array::Array{FT, 2},
    param_distribution::ParameterDistribution,
    param_slices::Array{Array{Int64, 1}, 1},
    param_dict::Dict,
    names::AbstractVector{String},
) where {FT}

    param_names_vec = typeof(names) <: AbstractVector ? names : [names]

    for (j, slice) in enumerate(param_slices)
        value = length(slice) > 1 ? param_array[slice, member] : param_array[slice, member][1]
        param_dict[param_names_vec[j]]["value"] = value
    end

    return param_dict
end


"""
$(TYPEDSIGNATURES)

Generate zero-padded subdirectory names of the form `<prefix>_<i>`.

# Arguments
- `N`: total number of members (when `mode = "all"`) or the specific member index (when `mode = "only"`).
- `prefix`: string prefix for directory names (default `"member"`).
- `mode`: `"all"` returns all names from 1 to `N`; `"only"` returns just the `N`th name.
- `pad_zeros`: number of digits for zero-padding indices.
"""
function generate_subdir_names(
    N::Int;
    prefix::AbstractString = "member",
    mode::AbstractString = "all",
    pad_zeros::Int = 3,
)


    member(j) = join([prefix, lpad(j, pad_zeros, "0")], "_")
    if mode == "all"
        return [member(j) for j in 1:N]
    else
        return member(N)
    end
end


"""
$(TYPEDSIGNATURES)

Return the names of all parameters in `param_dict` that are admissible for calibration.

A parameter is admissible when its sub-dictionary contains a `"prior"` key whose value
is not `"fixed"`, which allows non-UQ parameters to coexist in the same TOML file.

# Arguments
- `param_dict`: nested dictionary mapping parameter names to sub-dictionaries of parameter metadata.
"""
function get_admissible_parameters(param_dict::Dict)

    uq_param = String[]

    for (key, val) in param_dict
        if haskey(val, "prior") && val["prior"] != "fixed"
            push!(uq_param, string(key))
        end
    end

    return uq_param
end


"""
$(TYPEDSIGNATURES)

Write `param_dict` to a TOML file at `file_path`.

# Arguments
- `param_dict`: nested dictionary mapping parameter names to sub-dictionaries of parameter metadata.
- `file_path`: path of the TOML file to write.
"""
function write_log_file(param_dict::Dict, file_path::AbstractString)
    open(file_path, "w") do io
        TOML.print(io, param_dict)
    end
end


"""
$(TYPEDSIGNATURES)

Return the regularization type and coefficient for parameter `name` in `param_dict`.

Returns a tuple `(type, value)` where `type` is `"L1"` or `"L2"` and `value` is the
corresponding coefficient. Returns `(nothing, nothing)` when no regularization is specified.

# Arguments
- `param_dict`: nested dictionary mapping parameter names to sub-dictionaries of parameter metadata.
- `name`: name of the parameter to query.
"""
function get_regularization(param_dict::Dict, name::AbstractString)

    if haskey(param_dict[name], "L1") && haskey(param_dict[name], "L2")
        # There can't be more than one regularization flag
        throw(ArgumentError("Only one regularization flag (either \"L1\" or \"L2\") is allowed"))
    elseif haskey(param_dict[name], "L1")
        # L1 regularization
        return ("L1", param_dict[name]["L1"])
    elseif haskey(param_dict[name], "L2")
        # L1 regularization
        return ("L2", param_dict[name]["L2"])
    else
        # No regularization
        return (nothing, nothing)
    end
end


function get_regularization(param_dict::Dict, names::AbstractVector{String})

    regularr = []

    for name in names
        push!(regularr, get_regularization(param_dict, name))
    end

    return regularr
end

end # module
