# EKP implementation of the domain and observation localization from the literature

using ..ParameterDistributions
export UpdateGroup
export get_u_group, get_g_group, get_group_id, update_group_consistency, create_update_groups

"""
    struct UpdateGroup {VV <: AbstractVector}

Container of indices indicating which parameters (u_group) should be updated by which data (g_group).
Provide an array of UpdateGroups to partition the parameter space.
Note this partitioning assumes conditional independence between different sets of `UpdateGroups.u_group`s.

# Fields

$(TYPEDFIELDS)

"""
struct UpdateGroup
    "vector of parameter indices to form a partition of 1:input_dim) with other UpdateGroups provided"
    u_group::Vector{Int}
    "vector of data indices that lie within 1:output_dim)"
    g_group::Vector{Int}
    # process::Process # in future
    # localizer::Localizer # in future
    # inflation::Inflation # in future
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


get_u_group(group::UpdateGroup) = group.u_group
get_g_group(group::UpdateGroup) = group.g_group
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
            throw(
                ArgumentError(
                    "The combined 'UpdateGroup.u_group's must partition the indices of the input parameters: 1:$(input_dim), received: $(sort(u_flat))",
                ),
            )
        end
    end

    g_flat = reduce(vcat, g_groups)
    if eltype(g_flat) == Int
        if any(gf > output_dim for gf in g_flat) || any(gf <= 0 for gf in g_flat)
            throw(
                ArgumentError(
                    "The UpdateGroup.g_group must contains values in: 1:$(output_dim), found values outside this range",
                ),
            )
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
                throw(
                    ArgumentError(
                        "For group identifiers Dict(X => ...), X should be listed in $(param_names). Got $(pn).",
                    ),
                )
            end

            push!(u_group, isa(pi, Int) ? [pi] : pi)
        end
        for obn in val_vec
            oi = obs_indices[obn .== obs_names]
            if length(oi) == 0
                throw(
                    ArgumentError(
                        "For group identifiers Dict(... => Y), Y should be listed from $(obs_names). Instead got $(val).",
                    ),
                )
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
Base.:(==)(a::UG1, b::UG2) where {UG1 <: UpdateGroup, UG2 <: UpdateGroup} = all([
    get_u_group(a) == get_u_group(b),
    get_g_group(a) == get_g_group(b),
    get_group_id(a) == get_group_id(b),
],
                                                                                )

