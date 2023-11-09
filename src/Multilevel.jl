export SingleLevelScheduler, MultilevelScheduler, get_N_ens, get_N_indep, levels, transform_noise

struct LevelInfinity end

const SingleLevelType{IT} = Union{IT, LevelInfinity}

struct MultilevelScheduler{IT <: Integer} <: LevelScheduler
    Js::Dict{IT, IT}
    N_indep::IT
    N_ens::IT

    function MultilevelScheduler(Js::Dict{IT, IT}) where {IT <: Integer}
        N_indep = sum(values(Js))
        N_ens = sum(J * (level == 0 ? 1 : 2) for (level, J) in Js)

        new{IT}(Js, N_indep, N_ens)
    end
end

struct SingleLevelScheduler{IT <: Integer} <: LevelScheduler
    N_ens::IT
    level::SingleLevelType{IT}

    function SingleLevelScheduler(N_ens::IT, level::SingleLevelType{IT} = LevelInfinity()) where {IT <: Integer}
        new{IT}(N_ens, level)
    end
end


get_N_ens(ms::MultilevelScheduler) = ms.N_ens

get_N_indep(ms::MultilevelScheduler) = ms.N_indep

levels(ms::MultilevelScheduler) = begin
    vcat(
        fill(0, ms.Js[0]),
        (fill(l, ms.Js[l]) for l in sort(collect(keys(ms.Js))) if l != 0)...,
        (fill(l - 1, ms.Js[l]) for l in sort(collect(keys(ms.Js))) if l != 0)...,
    )
end

transform_noise(ms::MultilevelScheduler, noise::AbstractMatrix{FT}) where {FT <: Real} = begin
    @assert size(noise, 2) == ms.N_indep

    noise[:, vcat(1:ms.N_indep, ms.Js[0]+1:ms.N_indep)]
end

statistic_groups(ms::MultilevelScheduler) = begin
    groups = []

    offset = ms.N_indep - ms.Js[0]

    index = 0;
    for level in sort(collect(keys(ms.Js)))
        J = ms.Js[level]
        push!(groups, (index+1:index+J, 1))
        if level > 0
            push!(groups, (index+offset+1:index+offset+J, -1))
        end

        index += J
    end

    groups
end


get_N_ens(sls::SingleLevelScheduler) = sls.N_ens

get_N_indep(sls::SingleLevelScheduler) = sls.N_ens

levels(sls::SingleLevelScheduler) = fill(sls.level, sls.N_ens)

transform_noise(sls::SingleLevelScheduler, noise::AbstractMatrix{FT}) where {FT <: Real} = noise

statistic_groups(sls::SingleLevelScheduler) = [(1:sls.N_ens, 1)]
