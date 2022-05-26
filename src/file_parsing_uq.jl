using TOML
using Distributions
using EnsembleKalmanProcesses.ParameterDistributions


# "Public" functions (i.e., functions intended to be called by user):
#    - read_parameters
#    - get_parameter_distribution
#    - save_parameter_ensemble
#    - get_UQ_parameters
#    - get_regularization

"""
read_parameters(path_to_toml_file)

Read parameters from toml file

Args:
`path_to_toml_file` - path of the toml file containing the parameters to be
                      read.
                      See `CLIMAParameters/test/uq_test_parameters.toml` for
                      an example toml file that illustrates the expected
                      format of the parameter information.

Returns a nested dictionary whose keys are the parameter names (= headers of
the toml tables) and whose values are dictionaries containing the corresponding
parameter information (e.g., "prior", "constraint", "value", etc.)
"""
function read_parameters(path_to_toml_file::AbstractString)
    param_dict = TOML.parsefile(path_to_toml_file)
    return param_dict
end


"""
get_parameter_distribution(param_dict, name)

Constructs a `ParameterDistribution` for a single parameter

Args:
`param_dict` - nested dictionary that has parameter names as keys and the
               corresponding dictionary of parameter information (in particular,
               the parameters' prior distributions and constraints) as values
`name` - parameter name

Returns a `ParameterDistribution`
"""
function get_parameter_distribution(param_dict::Dict, name::AbstractString)

    # Constructing a parameter distribution requires a prior distribution,
    # a constraint, and a name.
    prior = construct_prior(param_dict[name])
    constraint = construct_constraint(param_dict[name])

    return ParameterDistribution(prior, constraint, name)
end

"""
get_parameter_distribution(param_dict, names)

Constructs a `ParameterDistribution` for an array of parameters

Args:
`param_dict` - nested dictionary that has parameter names as keys and the
               corresponding dictionary of parameter information (in particular,
               the parameters' prior distributions and constraints) as values
`names` - array of parameter names

Returns a `ParameterDistribution` 
"""
function get_parameter_distribution(param_dict::Dict, names::AbstractVector{String})
    param_dist_arr = map(names) do name
        get_parameter_distribution(param_dict, name)
    end
    return combine_distributions(param_dist_arr)
end


"""
construct_constraint(param_info)

Extracts information on type and arguments of each constraint and uses that
information to construct a `Constraint`.

Args:
`param_info` - dictionary with (at least) a key "constraint", whose value is
               the parameter's constraint(s) (as parsed from TOML file)

Returns a `Constraint`
"""
function construct_constraint(param_info::Dict)

    @assert(haskey(param_info, "constraint"))
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
construct_prior(param_info)

Extracts information on type and arguments of the prior distribution and use
that information to construct an actual `Distribution`

Args:
`param_info` - dictionary with (at least) a key "prior", whose value is the
               parameter's distribution(s) (as parsed from TOML file)

Returns a distribution of type `Parameterized`, `Samples`, or
`VectorOfParameterized`
"""
function construct_prior(param_info::Dict)

    @assert(haskey(param_info, "prior"))
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
get_vector_of_parameterized(d)

Parses a distribution of type `VectorOfParameterized`

Args:
`d`  - expression containing the distribution information

Returns a `VectorOfParameterized`
"""
function get_vector_of_parameterized(d::Expr)

    @assert(d.args[1] == Symbol("VectorOfParameterized"))

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
collect_from_expr(e, eltype; repeat=false)

Collects distributions or constraints

Args:
`e`  - expression containing the distribution or constraint information
`eltype` - string denoting the type of elements that are collected, "d" for
           distributions, "c" for constraints
`repeat` - true if this distribution or constraint is given as a `repeat(...)`
           expression, false otherwise

Returns an array of distributions / constraints, or a single distribution /
constraint if only one is present
"""
function collect_from_expr(e::Expr, eltype::AbstractString; repeat::Bool = false)

    if e.head == Symbol("vect")
        # There are multiple distributions / constraints 
        n_elem = length(e.args) # number of elements
        arr = (eltype == "d") ? Array{Distribution}(undef, n_elem) : Array{ConstraintType}(undef, n_elem)

        for i in 1:n_elem
            elem = e.args[i]
            arr[i] = getfield(Main, elem.args[1])(elem.args[2:end]...)
        end

        return repeat ? arr[1] : arr

    else
        # There is a single distribution / constraint
        return getfield(Main, e.args[1])(e.args[2:end]...)
    end

end


"""
get_distribution_from_expr(d)

Parses a distribution

Args:
`d`  - expression containing the distribution information

Returns a distribution of type `Parameterized` or `Samples`
"""
function get_distribution_from_expr(d::Expr)

    dist_type_symb = d.args[1]
    dist_type = getfield(Main, dist_type_symb)

    if dist_type_symb == Symbol("Parameterized")
        dist = getfield(Main, d.args[2].args[1])
        dist_args = d.args[2].args[2:end]

        return dist_type(dist(dist_args...))

    elseif dist_type_symb == Symbol("Samples")
        dist_args = construct_2d_array(d.args[2])

        return dist_type(dist_args)

    else
        throw(error("Unknown distribution type ", dist_type))
    end
end


"""
construct_2d_array(arr)

Reconstructs 2d array of samples

Args:
`arr`  - expression (has type `Expr`) with head `vcat`.

Returns a 2d array of samples constructed from the arguments of `expr`
"""
function construct_2d_array(arr::Expr)

    @assert(arr.head == Symbol("vcat"))
    n_rows = length(arr.args)
    arr_of_rows = [arr.args[i].args for i in 1:n_rows]

    return Float64.(vcat(arr_of_rows'...))
end


"""
save_parameter_ensemble(
    param_array,
    param_distribution,
    default_param_data,
    save_path,
    save_file,
    iter=nothing)

Saves the parameters in the given `param_array` to TOML files. The intended
use is for saving the ensemble of parameters after each update of an
ensemble Kalman process.
Each ensemble member (column of `param_array`) is saved in a separate
directory "member_<j>" (j=1, ..., N_ens). The name of the saved toml file is
given by `save_file`; it is the same for all members.
If an iteration `iter` is given, a directory "iteration_<iter>" is created in
`save_path`, which contains all the "member_<j>" subdirectories.

Args:
`param_array` - array of size N_param x N_ens
`param_distribution` - the parameter distribution underlying `param_array`
`default_param_data` - dict of default parameters to be combined and saved with
                       the parameters in `param_array` into a toml file
`save_path` - path to where the parameters will be saved
`save_file` - name of the toml files to be generated
`iter` - the iteration of the ensemble Kalman process represented by the given
         `param_array`
"""
function save_parameter_ensemble(
    param_array::Array{FT, 2},
    param_distribution::ParameterDistribution,
    default_param_data::Dict,
    save_path::AbstractString,
    save_file::AbstractString,
    iter::Union{Int, Nothing} = nothing,
) where {FT}

    # The parameter values are currently in the unconstrained space
    # where the ensemble Kalman algorithm takes place
    save_array = transform_unconstrained_to_constrained(param_distribution, param_array)

    # The number of rows in param_array represent the sum of all parameter
    # dimensions. We need to determine the slices of rows that belong to
    # each parameter. E.g., an array with 6 rows could be sliced into
    # one 1-dim parameter (first row), one 3-dim parameter (rows 2 to 4),
    # and a 2-dim parameter (rows 5 to 6)
    param_slices = batch(param_distribution)
    param_names = get_name(param_distribution)

    N_ens = size(save_array)[2]

    # Create directory where files will be stored if it doesn't exist yet
    save_dir = isnothing(iter) ? save_path : joinpath(save_path, join(["iteration", lpad(iter, 2, "0")], "_"))
    mkpath(save_dir)

    # Each ensemble member gets its own subdirectory
    subdir_names = generate_subdir_names(N_ens)

    # All parameter toml files (one for each ensemble member) have the same name
    toml_file = endswith(save_file, ".toml") ? save_file : save_file * ".toml"

    for i in 1:N_ens
        mkpath(joinpath(save_dir, subdir_names[i]))
        # Override the value (or add a value, if no value exists yet)
        # of the parameter in the original parameter dict with the
        # corresponding value in param_array
        param_dict = deepcopy(default_param_data)

        param_dict_updated = assign_values!(i, save_array, param_distribution, param_slices, param_dict, param_names)

        open(joinpath(save_dir, subdir_names[i], toml_file), "w") do io
            TOML.print(io, param_dict_updated)
        end
    end
end


"""
assign_values!(member, param_array, param_distribution, param_slices,
param_dict, names)

Updates `param_dict` with the values of the given `member` of the `param_array`

Args:
`member`  - ensemble member (corresponds to column of `param_array`)
`param_array` - N_par x N_ens array of parameter values
`param_distribution` - the parameter distribution underlying `param_array`
`param_slices` - list of contiguous `[collect(1:i), collect(i+1:j),... ]` used
                 to split parameter arrays by distribution dimensions
`param_dict` - the dict of parameters to be updated with new parameter values
`names` - array of parameter names

Returns the updated `param_dict`
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
generate_subdir_names(N_ens, prefix="member")

Generates `N_ens` directory names "<prefix>_<i>"; i=1, ..., N_ens

Args:
`N_ens`  - number of ensemble members (= number of subdirectories)
`prefix` - prefix used for generation of subdirectory names

Returns a list of directory names
"""
function generate_subdir_names(N_ens::Int, prefix::AbstractString = "member")

    member(j) = join([prefix, lpad(j, ndigits(N_ens), "0")], "_")

    return [member(j) for j in 1:N_ens]
end


"""
get_UQ_parameters(param_dict)

Finds all UQ parameters in `param_dict`.

Args:
`param_dict` - nested dictionary that has parameter names as keys and the
               corresponding dictionaries of parameter information as values

Returns an array of the names of all UQ parameters in `param_dict`.
UQ parameters are those parameters that have a key "prior" whose value is not
set to "fixed". They will enter the uncertainty quantification pipeline.
"""
function get_UQ_parameters(param_dict::Dict)

    uq_param = String[]

    for (key, val) in param_dict
        if haskey(val, "prior") && val["prior"] != "fixed"
            push!(uq_param, string(key))
        end
    end

    return uq_param
end


"""
write_log_file(param_dict, file_path)

Writes the parameters in `param_dict` into a .toml file

Args:
`param_dict` - nested dictionary that has parameter names as keys and the
               corresponding dictionaries of parameter information as values
`file_path` - path of the file where parameters are saved
"""
function write_log_file(param_dict::Dict, file_path::AbstractString) where {FT}
    open(file_path, "w") do io
        TOML.print(io, param_dict)
    end
end


"""
get_regularization(param_dict, name)

Returns the regularization information for a single parameter

Args:
`param_dict` - nested dictionary that has parameter names as keys and the
               corresponding dictionary of parameter information as values
`name` - parameter name

Returns a tuple (<regularization_type>, <regularization_value>), where the
regularization type is either "L1" or "L2", and the regularization value is
a float. Returns (nothing, nothing) if parameter has no regularization
information.
"""
function get_regularization(param_dict::Dict, name::AbstractString)

    if haskey(param_dict[name], "L1") && haskey(param_dict[name], "L2")
        # There can't be more than one regularization flag
        throw(ErrorException("Only one regularization flag (either 'L1' or " * "'L2') is allowed"))
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


"""
get_regularization(param_dict, names)

Returns the regularization information for an array of parameters

Args:
`param_dict` - nested dictionary that has parameter names as keys and the
               corresponding dictionary of parameter information as values
`names` - array of parameter names

Returns an arary of tuples (<regularization_type>, <regularization_value>), with
the ith tuple corresponding to the parameter `names[i]`. The regularization
type is either "L1" or "L2", and the regularization value is a float.
Returns (nothing, nothing) for parameters that have no regularization
information.
"""
function get_regularization(param_dict::Dict, names::AbstractVector{String})

    regularr = []

    for name in names
        push!(regularr, get_regularization(param_dict, name))
    end

    return regularr
end
