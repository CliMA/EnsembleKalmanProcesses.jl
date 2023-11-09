# included in EnsembleKalmanProcess.jl

export compute_mean, compute_cov

function posdef(mat)
    S, V = eigen(mat)
    V = V[:, (S .> 0)]
    S = S[S .> 0]
    V * diagm(S) * V'
end

function compute_mean(ekp::EnsembleKalmanProcess, x; ignored_indices = [])
    reduce(statistic_groups(ekp.level_scheduler); init = 0) do acc, (indices, multiplier)
        indices = setdiff(indices, ignored_indices)
        multiplier * mean(x[:, indices]; dims = 2) .+ acc
    end
end

function compute_cov(ekp::EnsembleKalmanProcess, x; corrected, ignored_indices = [])
    reduce(statistic_groups(ekp.level_scheduler); init = 0) do acc, (indices, multiplier)
        indices = setdiff(indices, ignored_indices)
        multiplier * cov(x[:, indices]; corrected, dims = 2) .+ acc
    end
end
