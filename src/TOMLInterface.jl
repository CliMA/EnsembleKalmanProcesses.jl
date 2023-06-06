module TOMLInterface

# Imports
using TOML
using Distributions
using EnsembleKalmanProcesses.ParameterDistributions


# Exports
export path_to_ensemble_member
export get_parameter_distribution
export get_parameter_values
export save_parameter_ensemble
export get_admissible_parameters
export get_regularization
export write_log_file


"""
    get_parameter_values(param_dict, names)

Gets parameter values from a parameter dictionary, indexing by name.

Args:
`param_dict` - nested dictionary that has parameter names as keys and the
               corresponding dictionary of parameter information (in particular,
               the parameters' values)
`name` - iterable parameter names
`return_type` - return type, default "dict", otherwise "array"
"""
function get_parameter_values(param_dict::Dict, names; return_type = "dict")
    if return_type == "dict"
        return Dict(n => param_dict[n]["value"] for n in names)
    elseif return_type == "array"
        return [param_dict[n]["value"] for n in names]
    else
        throw(ArgumentError("Unknown `return_type`. Expected \"dict\" or \"array\", got $return_type"))
    end
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

    if dist_type_symb == Symbol("Parameterized")
        dist = getfield(Main, d.args[2].args[1])
        dist_args = d.args[2].args[2:end]
        dist_type = getfield(Main, dist_type_symb)

        return dist_type(dist(dist_args...))

    elseif dist_type_symb == Symbol("Samples")
        dist_args = construct_2d_array(d.args[2])
        dist_type = getfield(Main, dist_type_symb)

        return dist_type(dist_args)

    else
        throw(ArgumentError("Unknown distribution type from symbol: $(dist_type_symb)"))
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
        iteration
        pad_zeros=3,
    apply_constraints=true
    )

Saves the parameters in the given `param_array` to TOML files. The intended
use is for saving the ensemble of parameters after each update of an
ensemble Kalman process.
Each ensemble member (column of `param_array`) is saved in a separate
directory "member_<j>" (j=1, ..., N_ens). The name of the saved toml file is
given by `save_file`; it is the same for all members.
A directory "iteration_<iter>" is created in `save_path`, which contains all the "member_<j>" subdirectories.

Args:
`param_array` - array of size N_param x N_ens
`param_distribution` - the parameter distribution underlying `param_array`
`apply_constraints` -  apply the constraints in `param_distribution`
`default_param_data` - dict of default parameters to be combined and saved with
                       the parameters in `param_array` into a toml file
`save_path` - path to where the parameters will be saved
`save_file` - name of the toml files to be generated
`iteration` - the iteration of the ensemble Kalman process represented by the given
         `param_array`
`pad_zeros` - the amount of zero-padding for the ensemble member number
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

"""
One can also call this without the iteration level
"""
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
    path_to_ensemble_member(
        base_path,
        iteration,
        member,
        pad_zeros = 3,
    )

Obtains the file path to a specified ensemble member. The likely form is
`base_path/iteration_X/member_Y/` with X,Y padded with zeros. The file path can be reconstructed with:
`base_path` - base path to where EKP parameters are stored
`member` - number of the ensemble member (without zero padding)
`iteration` - iteration of ensemble method (if =nothing then only the load path is used)
`pad_zeros` - amount of digits to pad to
"""
function path_to_ensemble_member(base_path::AbstractString, iteration::Int, member::Int; pad_zeros = 3)

    # Get the directory of the iteration
    base_dir = joinpath(base_path, join(["iteration", lpad(iteration, pad_zeros, "0")], "_"))
    return path_to_ensemble_member(base_dir, member, pad_zeros = pad_zeros)
end

"""
One can also call this without the iteration level
"""
function path_to_ensemble_member(base_path::AbstractString, member::Int; pad_zeros = 3)
    # get the directory of the member
    subdir_name = generate_subdir_names(member, mode = "only", pad_zeros = pad_zeros)
    return joinpath(base_path, subdir_name)
end

"""
    assign_values!(
        member, 
        param_array, 
        param_distribution, 
        param_slices, 
        param_dict, 
        names)

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
    generate_subdir_names(N; prefix="member", mode="all", pad_zeros=3)

Generates `N` directory names "<prefix>_<i>"; i=1, ..., N

Args:
`N`  - number of ensemble members (= number of subdirectories) or for `mode=only`, the chosen member
`prefix` - prefix used for generation of subdirectory names
`mode`   - default `=all` generates all names, `=only` generates just the `N`th name
`pad_zeros` - amount of digits to pad to
Returns a list of directory names
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
    get_admissible_parameters(param_dict)

Finds all parameters in `param_dict` that are admissible for calibration.

Args:
`param_dict` - nested dictionary that has parameter names as keys and the
               corresponding dictionaries of parameter information as values

Returns an array of the names of all admissible parameters in `param_dict`.
Admissible parameters must have a key "prior" and the value value of this is not
set to "fixed". This allows for other parameters to be stored within the TOML file.
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
    write_log_file(param_dict, file_path)

Writes the parameters in `param_dict` into a .toml file

Args:
`param_dict` - nested dictionary that has parameter names as keys and the
               corresponding dictionaries of parameter information as values
`file_path` - path of the file where parameters are saved
"""
function write_log_file(param_dict::Dict, file_path::AbstractString)
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


"""
    get_regularization(param_dict, names)

Returns the regularization information for an array of parameters

Args:
`param_dict` - nested dictionary that has parameter names as keys and the
               corresponding dictionary of parameter information as values
`names` - array of parameter names

Returns an arary of tuples (<regularization_type>, <regularization_value>), with the ith tuple corresponding to the parameter `names[i]`. 
The regularization type is either "L1" or "L2", and the regularization 
value is a float.
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

end # module
