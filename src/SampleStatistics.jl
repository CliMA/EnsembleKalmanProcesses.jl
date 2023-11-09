# included in EnsembleKalmanProcess.jl

export compute_mean, compute_cov

function posdef(mat)
    S, V = eigen(mat)
    V = V[:, (S .> 0)]
    S = S[S .> 0]
    V * diagm(S) * V'
end

function compute_mean(ekp::EnsembleKalmanProcess, u)
    foldl(statistic_groups(ekp.level_scheduler); init = 0) do acc, (indices, multiplier)
        multiplier * mean(u[:, indices]; dims = 2) .+ acc
    end
end

function compute_cov(ekp::EnsembleKalmanProcess, u, g; corrected)
    foldl(statistic_groups(ekp.level_scheduler); init = 0) do acc, (indices, multiplier)
        multiplier * cov([u; g][:, indices]; corrected = false, dims = 2) .+ acc
    end
end
