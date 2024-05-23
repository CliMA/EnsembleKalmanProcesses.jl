# EKP implementation of the domain and observation localization from the literature

export UpdateGroup
export get_u_group, get_g_group, update_group_consistency

"""
    struct UpdateGroup {VV <: AbstractVector}

Container of indices indicating which parameters (u_group) should be updated by which data (g_group).
Provide an array of UpdateGroups to partition the parameter space.
Note this partitioning assumes conditional independence between different sets of `UpdateGroups.u_group`s.

# Fields

$(TYPEDFIELDS)

"""
struct UpdateGroup
    "vector of parameter indices, forms part(or all) of a partition of 1:input_dim with other UpdateGroups provided"
    u_group::Vector{Int}
    "vector of data indices, must lie within 1:output_dim"
    g_group::Vector{Int}
    # process::Process # in future
    # localizer::Localizer # in future
    # inflation::Inflation # in future
end

get_u_group(group::UpdateGroup) = group.u_group
get_g_group(group::UpdateGroup) = group.g_group

function get_u_group(groups::VV) where {VV <: AbstractVector}
    u_group = []
    for group in groups
        push!(u_group, get_u_group(group))
    end
    return u_group
end

function get_g_group(groups::VV) where {VV <: AbstractVector}
    g_group = []
    for group in groups
        push!(g_group, get_g_group(group))
    end
    return g_group
end

# check an array of update_groups are consistent, i.e. common sizing for u,g, and that u is a partition.
"""
$(TYPEDSIGNATURES)

Check the consistency of sizing and partitioning of the `UpdateGroup` array
"""
function update_group_consistency(groups::VV, input_dim::Int, output_dim::Int) where {VV <: AbstractVector}

    u_groups = get_u_group(groups)
    g_groups = get_g_group(groups)

    # check there is an index in each group
    if any(length(group) == 0 for group in u_groups)
        throw(ArgumentError("all `UpdateGroup.u_group` must contain at least one parameter index"))
    end
    if any(length(group) == 0 for group in g_groups)
        throw(ArgumentError("all `UpdateGroup.g_group` must contain at least one parameter index"))
    end

    # check for partition
    u_flat = reduce(vcat, u_groups)
    if !(1:input_dim == sort(u_flat))
        throw(
            ArgumentError(
                "The combined 'UpdateGroup.u_group's must partition the indices of the input parameters: 1:$(input_dim), received: $(sort(u_flat))",
            ),
        )
    end

    g_flat = reduce(vcat, g_groups)
    if any(gf > output_dim for gf in g_flat) || any(gf <= 0 for gf in g_flat)
        throw(
            ArgumentError(
                "The UpdateGroup.g_group must contains values in: 1:$(output_dim), found values outside this range",
            ),
        )
    end

    # pass the tests
    return true
end
