# EKP implementation of the domain and observation localization from the literature

using ..ParameterDistributions
export UpdateGroup
export get_u_group, get_g_group, get_group_id, update_group_consistency, create_update_groups

"""
Container of indices indicating which parameters (`u_group`) are updated using which data (`g_group`).

Provide an array of `UpdateGroup`s to partition the parameter space. This partitioning assumes
conditional independence between the different `u_group`s.

$(TYPEDEF)

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
struct UpdateGroup
    "vector of parameter indices forming a partition of `1:input_dim` with other `UpdateGroup`s"
    u_group::Vector{Int}
    "vector of data indices within `1:output_dim`"
    g_group::Vector{Int}
    # process::Process # in future
    # localizer::Localizer # in future
    # inflation::Inflation # in future
    "mapping of parameter index range to data index range, used for group identification"
    group_id::Dict
end

function UpdateGroup(u_group::VV1, g_group::VV2) where {VV1 <: AbstractVector, VV2 <: AbstractVector}
    # check there is an index in each group
    if length(u_group) == 0
        throw(ArgumentError("all `UpdateGroup.u_group` must contain at least one parameter identifier"))
    end
    if length(g_group) == 0
        throw(ArgumentError("all `UpdateGroup.g_group` must contain at least one data identifier"))
    end
    return UpdateGroup(
        u_group,
        g_group,
        Dict("[$(minimum(u_group)),...,$(maximum(u_group))]" => "[$(minimum(g_group)),...,$(maximum(g_group))]"),
    )
end


"""
$(TYPEDSIGNATURES)

Return the parameter index vector stored in `group`.
"""
get_u_group(group::UpdateGroup) = group.u_group

"""
$(TYPEDSIGNATURES)

Return the data index vector stored in `group`.
"""
get_g_group(group::UpdateGroup) = group.g_group

"""
$(TYPEDSIGNATURES)

Return the group identification dictionary stored in `group`.
"""
get_group_id(group::UpdateGroup) = group.group_id



# check an array of update_groups are consistent, i.e. common sizing for u,g, and that u is a partition.
"""
$(TYPEDSIGNATURES)

Check the consistency of sizing and partitioning of the `UpdateGroup` array if it contains indices
No consistency check if u,g has strings internally
"""
function update_group_consistency(groups::VV, input_dim::Int, output_dim::Int) where {VV <: AbstractVector}

    u_groups = get_u_group.(groups)
    g_groups = get_g_group.(groups)

    # check for partition (only if indices passed)
    u_flat = reduce(vcat, u_groups)
    if !(1:input_dim == sort(u_flat))
        if eltype(sort(u_flat)) == Int
            throw(ArgumentError("""
UpdateGroup u_group indices do not form a partition of the input parameter indices.

Expected:
    union of all u_groups == 1:$(input_dim)

Got:
    $(sort(u_flat))
"""))
        end
    end

    g_flat = reduce(vcat, g_groups)
    if eltype(g_flat) == Int
        if any(gf > output_dim for gf in g_flat) || any(gf <= 0 for gf in g_flat)
            bad = filter(gf -> gf > output_dim || gf <= 0, g_flat)
            throw(ArgumentError("""
UpdateGroup g_group indices are out of range.

Expected:
    all values in 1:$(output_dim)

Got:
    out-of-range values = $(sort(bad))
"""))
        end
    end
    # pass the tests
    return true
end

## Convience constructor for update_groups
"""
$(TYPEDSIGNATURES)

To construct a list of update-groups populated by indices of parameter distributions and indices of observations, from a dictionary of `group_identifiers = Dict(..., group_k_input_names => group_k_output_names, ...)`
"""
function create_update_groups(
    prior::PD,
    observation::OB,
    group_identifiers::Dict,
) where {PD <: ParameterDistribution, OB <: Observation}

    param_names = get_name(prior)
    param_indices = batch(prior, function_parameter_opt = "dof")

    obs_names = get_names(observation)
    obs_indices = get_indices(observation)

    update_groups = []
    for (key, val) in pairs(group_identifiers)
        key_vec = isa(key, AbstractString) ? [key] : key # make it iterable
        val_vec = isa(val, AbstractString) ? [val] : val

        u_group = []
        g_group = []
        for pn in key_vec
            pi = param_indices[pn .== param_names]
            if length(pi) == 0
                _throw_ug_bad_param_name(pn, param_names, key, key_vec)
            end

            push!(u_group, isa(pi, Int) ? [pi] : pi)
        end
        for obn in val_vec
            oi = obs_indices[obn .== obs_names]
            if length(oi) == 0
                _throw_ug_bad_obs_name(obn, obs_names, key, val_vec)
            end
            push!(g_group, isa(oi, Int) ? [oi] : oi)
        end
        u_group = reduce(vcat, reduce(vcat, u_group))
        g_group = reduce(vcat, reduce(vcat, g_group))
        push!(update_groups, UpdateGroup(u_group, g_group, Dict(key_vec => val_vec)))
    end
    return update_groups

end

## Overload ==
Base.:(==)(a::UG1, b::UG2) where {UG1 <: UpdateGroup, UG2 <: UpdateGroup} =
    all([get_u_group(a) == get_u_group(b), get_g_group(a) == get_g_group(b), get_group_id(a) == get_group_id(b)],)

## Error helpers

@noinline function _throw_ug_bad_param_name(pn, param_names, key, key_vec)
    throw(ArgumentError("""
Unrecognized parameter name in group_identifiers key.

Expected:
    a name from the prior's parameter names: $param_names

Got:
    $(repr(pn))

Loop context:
    group_identifiers key being processed: $(repr(key))
    full parameter name list for this key:  $key_vec

Suggestion:
    Check spelling and ensure the parameter name matches one returned by get_name(prior).
"""))
end

@noinline function _throw_ug_bad_obs_name(obn, obs_names, key, val_vec)
    throw(ArgumentError("""
Unrecognized observation name in group_identifiers value.

Expected:
    a name from the observation's names: $obs_names

Got:
    $(repr(obn))

Loop context:
    group_identifiers key this value belongs to: $(repr(key))
    full observation name list for this value:   $val_vec

Suggestion:
    Check spelling and ensure the observation name matches one returned by get_names(observation).
"""))
end
