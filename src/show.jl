# Base.show and Base.summary for all major EnsembleKalmanProcesses types.

# ── DataContainers ───────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::DataContainers.DataContainer)
    if get(io, :compact, false)
        show(io, x)
    else
        m, n = size(x.data)
        println(io, "DataContainer")
        println(io, "  size: ", m, " × ", n)
    end
end

function Base.show(io::IO, x::DataContainers.DataContainer)
    m, n = size(x.data)
    print(io, "DataContainer (", m, "×", n, ")")
end

function Base.summary(io::IO, x::DataContainers.DataContainer)
    m, n = size(x.data)
    print(io, "DataContainer (", m, "×", n, ")")
end

function Base.show(io::IO, ::MIME"text/plain", x::DataContainers.PairedDataContainer)
    if get(io, :compact, false)
        show(io, x)
    else
        m_in, n_in = size(x.inputs.data)
        m_out, n_out = size(x.outputs.data)
        println(io, "PairedDataContainer")
        println(io, "  inputs : ", m_in, " × ", n_in, " params × samples")
        println(io, "  outputs: ", m_out, " × ", n_out, " obs × samples")
    end
end

function Base.show(io::IO, x::DataContainers.PairedDataContainer)
    m_in, n_in = size(x.inputs.data)
    m_out, n_out = size(x.outputs.data)
    print(io, "PairedDataContainer (", m_in, "×", n_in, " → ", m_out, "×", n_out, ")")
end

function Base.summary(io::IO, x::DataContainers.PairedDataContainer)
    m_in, n_in = size(x.inputs.data)
    m_out, n_out = size(x.outputs.data)
    print(io, "PairedDataContainer (", m_in, "×", n_in, " → ", m_out, "×", n_out, ")")
end

# ── Observations: covariance helpers ─────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::SVDplusD)
    if get(io, :compact, false)
        show(io, x)
    else
        n = size(x.diag_cov, 1)
        println(io, "SVDplusD")
        println(io, "  size: ", n, " × ", n)
        println(io, "  rank: ", length(x.svd_cov.S))
    end
end

function Base.show(io::IO, x::SVDplusD)
    n = size(x.diag_cov, 1)
    print(io, "SVDplusD (", n, "×", n, ", rank ", length(x.svd_cov.S), ")")
end

function Base.summary(io::IO, x::SVDplusD)
    n = size(x.diag_cov, 1)
    print(io, "SVDplusD (", n, "×", n, ", rank ", length(x.svd_cov.S), ")")
end

function Base.show(io::IO, ::MIME"text/plain", x::DminusTall)
    if get(io, :compact, false)
        show(io, x)
    else
        n = size(x.diag_cov, 1)
        println(io, "DminusTall")
        println(io, "  size: ", n, " × ", n)
    end
end

function Base.show(io::IO, x::DminusTall)
    n = size(x.diag_cov, 1)
    print(io, "DminusTall (", n, "×", n, ")")
end

function Base.summary(io::IO, x::DminusTall)
    n = size(x.diag_cov, 1)
    print(io, "DminusTall (", n, "×", n, ")")
end

# ── Observations: Observation ─────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::Observation)
    if get(io, :compact, false)
        show(io, x)
    else
        n_blocks = length(x.samples)
        total_dim = isempty(x.indices) ? 0 : last(last(x.indices))
        println(io, "Observation")
        println(io, "  n_blocks : ", n_blocks)
        println(io, "  total_dim: ", total_dim)
        println(io, "  metadata : ", isnothing(x.metadata) ? "nothing" : typeof(x.metadata))
    end
end

function Base.show(io::IO, x::Observation)
    n_blocks = length(x.samples)
    total_dim = isempty(x.indices) ? 0 : last(last(x.indices))
    print(io, "Observation (", n_blocks, " block", n_blocks == 1 ? "" : "s", ", dim=", total_dim, ")")
end

function Base.summary(io::IO, x::Observation)
    n_blocks = length(x.samples)
    total_dim = isempty(x.indices) ? 0 : last(last(x.indices))
    print(io, "Observation (", n_blocks, " block", n_blocks == 1 ? "" : "s", ", dim=", total_dim, ")")
end

# ── Observations: Minibatchers ────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::FixedMinibatcher)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "FixedMinibatcher")
        println(io, "  n_batches: ", length(x.minibatches))
        println(io, "  method   : ", x.method)
    end
end

function Base.show(io::IO, x::FixedMinibatcher)
    print(io, "FixedMinibatcher (", length(x.minibatches), " batches, \"", x.method, "\")")
end

function Base.summary(io::IO, x::FixedMinibatcher)
    print(io, "FixedMinibatcher (", length(x.minibatches), " batches, \"", x.method, "\")")
end

function Base.show(io::IO, ::MIME"text/plain", x::RandomFixedSizeMinibatcher)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "RandomFixedSizeMinibatcher")
        println(io, "  batch_size: ", x.minibatch_size)
        println(io, "  method    : ", x.method)
    end
end

function Base.show(io::IO, x::RandomFixedSizeMinibatcher)
    print(io, "RandomFixedSizeMinibatcher (size=", x.minibatch_size, ", \"", x.method, "\")")
end

function Base.summary(io::IO, x::RandomFixedSizeMinibatcher)
    print(io, "RandomFixedSizeMinibatcher (size=", x.minibatch_size, ", \"", x.method, "\")")
end

# ── Observations: ObservationSeries ──────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::ObservationSeries)
    if get(io, :compact, false)
        show(io, x)
    else
        n_obs = length(x.observations)
        n_epochs = length(x.minibatches)
        println(io, "ObservationSeries")
        println(io, "  n_observations: ", n_obs)
        println(io, "  n_epochs      : ", n_epochs)
        println(io, "  minibatcher   : ", nameof(typeof(x.minibatcher)))
        println(io, "  metadata      : ", isnothing(x.metadata) ? "nothing" : typeof(x.metadata))
    end
end

function Base.show(io::IO, x::ObservationSeries)
    n_obs = length(x.observations)
    print(io, "ObservationSeries (", n_obs, " observation", n_obs == 1 ? "" : "s", ")")
end

function Base.summary(io::IO, x::ObservationSeries)
    n_obs = length(x.observations)
    print(io, "ObservationSeries (", n_obs, " observation", n_obs == 1 ? "" : "s", ")")
end

# ── ParameterDistributions: ParameterDistribution ────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::ParameterDistributions.ParameterDistribution)
    if get(io, :compact, false)
        show(io, x)
    else
        n = length(x.name)
        println(io, "ParameterDistribution with ", n, " entr", n == 1 ? "y" : "ies")
        max_show = 8
        for (i, inds) in enumerate(ParameterDistributions.batch(x, function_parameter_opt = "constraint"))
            i > max_show && break
            n_con = length(inds)
            println(
                io,
                "  '",
                x.name[i],
                "': ",
                sprint(summary, x.distribution[i]),
                " [",
                n_con,
                " constraint",
                n_con == 1 ? "" : "s",
                "]",
            )
        end
        n > max_show && println(io, "  … and ", n - max_show, " more")
    end
end

function Base.show(io::IO, x::ParameterDistributions.ParameterDistribution)
    n = length(x.name)
    print(io, "ParameterDistribution (", n, " entr", n == 1 ? "y" : "ies", ")")
end

function Base.summary(io::IO, x::ParameterDistributions.ParameterDistribution)
    n = length(x.name)
    print(io, "ParameterDistribution (", n, " entr", n == 1 ? "y" : "ies", ")")
end

# ── ParameterDistributions: Constraint ───────────────────────────────────────

function Base.show(
    io::IO,
    ::MIME"text/plain",
    cons::ParameterDistributions.Constraint{T},
) where {T <: ParameterDistributions.BasicConstraints}
    if get(io, :compact, false)
        show(io, cons)
    else
        bounds = isnothing(cons.bounds) ? Dict() : cons.bounds
        lb = get(bounds, "lower_bound", "-∞")
        ub = get(bounds, "upper_bound", "∞")
        print(io, "Constraint{$(T)} with bounds ($(lb), $(ub))")
    end
end

function Base.show(io::IO, cons::ParameterDistributions.Constraint{T}) where {T}
    suffix = isnothing(cons.bounds) ? "" : " with characterization $(tuple(cons.bounds...))"
    print(io, "Constraint{$(T)}" * suffix)
end

function Base.show(io::IO, cons::ParameterDistributions.Constraint{<:ParameterDistributions.BasicConstraints})
    bounds = isnothing(cons.bounds) ? Dict() : cons.bounds
    lb = get(bounds, "lower_bound", "-∞")
    ub = get(bounds, "upper_bound", "∞")
    print(io, "Bounds: ($(lb), $(ub))")
end

function Base.summary(
    io::IO,
    cons::ParameterDistributions.Constraint{T},
) where {T <: ParameterDistributions.BasicConstraints}
    bounds = isnothing(cons.bounds) ? Dict() : cons.bounds
    lb = get(bounds, "lower_bound", "-∞")
    ub = get(bounds, "upper_bound", "∞")
    print(io, "Constraint{$(T)} ($(lb), $(ub))")
end

function Base.summary(io::IO, cons::ParameterDistributions.Constraint{T}) where {T}
    print(io, "Constraint{$(T)}")
end

# ── ParameterDistributions: Parameterized ────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::ParameterDistributions.Parameterized)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "Parameterized")
        println(io, "  distribution: ", nameof(typeof(x.distribution)))
    end
end

function Base.show(io::IO, x::ParameterDistributions.Parameterized)
    print(io, "Parameterized (", nameof(typeof(x.distribution)), ")")
end

function Base.summary(io::IO, x::ParameterDistributions.Parameterized)
    print(io, "Parameterized (", nameof(typeof(x.distribution)), ")")
end

# ── ParameterDistributions: Samples ──────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::ParameterDistributions.Samples)
    if get(io, :compact, false)
        show(io, x)
    else
        m, n = size(x.distribution_samples)
        println(io, "Samples")
        println(io, "  size: ", m, " × ", n, "  (", m, " params, ", n, " samples)")
    end
end

function Base.show(io::IO, x::ParameterDistributions.Samples)
    m, n = size(x.distribution_samples)
    print(io, "Samples (", m, "×", n, ")")
end

function Base.summary(io::IO, x::ParameterDistributions.Samples)
    m, n = size(x.distribution_samples)
    print(io, "Samples (", m, "×", n, ")")
end

# ── ParameterDistributions: VectorOfParameterized ────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::ParameterDistributions.VectorOfParameterized)
    if get(io, :compact, false)
        show(io, x)
    else
        n = length(x.distribution)
        println(io, "VectorOfParameterized")
        println(io, "  n_distributions: ", n)
        if n > 0
            println(io, "  eltype         : ", eltype(x.distribution))
        end
    end
end

function Base.show(io::IO, x::ParameterDistributions.VectorOfParameterized)
    print(io, "VectorOfParameterized (", length(x.distribution), " distributions)")
end

function Base.summary(io::IO, x::ParameterDistributions.VectorOfParameterized)
    print(io, "VectorOfParameterized (", length(x.distribution), " distributions)")
end

# ── ParameterDistributions: GaussianRandomFieldInterface ─────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::ParameterDistributions.GaussianRandomFieldInterface)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "GaussianRandomFieldInterface")
        println(io, "  package  : ", nameof(typeof(x.package)))
        println(io, "  n_dofs   : ", length(x.distribution.name))
    end
end

function Base.show(io::IO, x::ParameterDistributions.GaussianRandomFieldInterface)
    print(io, "GaussianRandomFieldInterface (", nameof(typeof(x.package)), ", ", length(x.distribution.name), " dofs)")
end

function Base.summary(io::IO, x::ParameterDistributions.GaussianRandomFieldInterface)
    print(io, "GaussianRandomFieldInterface (", nameof(typeof(x.package)), ", ", length(x.distribution.name), " dofs)")
end

# ── UpdateGroup ───────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::UpdateGroup)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "UpdateGroup")
        println(io, "  |u_group|: ", length(x.u_group))
        println(io, "  |g_group|: ", length(x.g_group))
    end
end

function Base.show(io::IO, x::UpdateGroup)
    print(io, "UpdateGroup (|u|=", length(x.u_group), " → |g|=", length(x.g_group), ")")
end

function Base.summary(io::IO, x::UpdateGroup)
    print(io, "UpdateGroup (|u|=", length(x.u_group), " → |g|=", length(x.g_group), ")")
end

# ── Accelerators ──────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::ConstantNesterovAccelerator)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "ConstantNesterovAccelerator")
        println(io, "  λ         : ", x.λ)
        println(io, "  state_size: ", size(x.u_prev))
    end
end

function Base.show(io::IO, x::ConstantNesterovAccelerator)
    print(io, "ConstantNesterovAccelerator (λ=", x.λ, ")")
end

function Base.summary(io::IO, x::ConstantNesterovAccelerator)
    print(io, "ConstantNesterovAccelerator (λ=", x.λ, ")")
end

function Base.show(io::IO, ::MIME"text/plain", x::FirstOrderNesterovAccelerator)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "FirstOrderNesterovAccelerator")
        println(io, "  r         : ", x.r)
        println(io, "  state_size: ", size(x.u_prev))
    end
end

function Base.show(io::IO, x::FirstOrderNesterovAccelerator)
    print(io, "FirstOrderNesterovAccelerator (r=", x.r, ")")
end

function Base.summary(io::IO, x::FirstOrderNesterovAccelerator)
    print(io, "FirstOrderNesterovAccelerator (r=", x.r, ")")
end

function Base.show(io::IO, ::MIME"text/plain", x::NesterovAccelerator)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "NesterovAccelerator")
        println(io, "  θ_prev    : ", x.θ_prev)
        println(io, "  state_size: ", size(x.u_prev))
    end
end

function Base.show(io::IO, x::NesterovAccelerator)
    print(io, "NesterovAccelerator (θ_prev=", x.θ_prev, ")")
end

function Base.summary(io::IO, x::NesterovAccelerator)
    print(io, "NesterovAccelerator (θ_prev=", x.θ_prev, ")")
end

# ── LearningRateSchedulers ────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::DataMisfitController)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "DataMisfitController")
        println(io, "  terminate_at: ", x.terminate_at)
        println(io, "  on_terminate: ", x.on_terminate)
        println(io, "  n_iterations: ", length(x.iteration))
    end
end

function Base.show(io::IO, x::DataMisfitController)
    print(io, "DataMisfitController (T=", x.terminate_at, ", \"", x.on_terminate, "\")")
end

function Base.summary(io::IO, x::DataMisfitController)
    print(io, "DataMisfitController (T=", x.terminate_at, ", \"", x.on_terminate, "\")")
end

# ── Process types ─────────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::Inversion)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "Inversion")
        println(io, "  impose_prior                   : ", x.impose_prior)
        println(io, "  default_multiplicative_inflation: ", x.default_multiplicative_inflation)
        if !isnothing(x.prior_mean)
            println(io, "  prior_dim: ", length(x.prior_mean))
        end
    end
end

function Base.show(io::IO, x::Inversion)
    print(io, "Inversion", x.impose_prior ? " (with prior)" : "")
end

function Base.summary(io::IO, x::Inversion)
    print(io, "Inversion", x.impose_prior ? " (with prior)" : "")
end

function Base.show(io::IO, ::MIME"text/plain", x::TransformInversion)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "TransformInversion")
        println(io, "  impose_prior                   : ", x.impose_prior)
        println(io, "  default_multiplicative_inflation: ", x.default_multiplicative_inflation)
        if !isnothing(x.prior_mean)
            println(io, "  prior_dim: ", length(x.prior_mean))
        end
    end
end

function Base.show(io::IO, x::TransformInversion)
    print(io, "TransformInversion", x.impose_prior ? " (with prior)" : "")
end

function Base.summary(io::IO, x::TransformInversion)
    print(io, "TransformInversion", x.impose_prior ? " (with prior)" : "")
end

function Base.show(io::IO, ::MIME"text/plain", x::Sampler)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "Sampler{", nameof(get_sampler_type(x)), "}")
        println(io, "  prior_dim: ", length(x.prior_mean))
    end
end

function Base.show(io::IO, x::Sampler)
    print(io, "Sampler{", nameof(get_sampler_type(x)), "} (prior_dim=", length(x.prior_mean), ")")
end

function Base.summary(io::IO, x::Sampler)
    print(io, "Sampler{", nameof(get_sampler_type(x)), "} (prior_dim=", length(x.prior_mean), ")")
end

function Base.show(io::IO, ::MIME"text/plain", x::GaussNewtonInversion)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "GaussNewtonInversion")
        println(io, "  prior_dim: ", length(x.prior_mean))
    end
end

function Base.show(io::IO, x::GaussNewtonInversion)
    print(io, "GaussNewtonInversion (prior_dim=", length(x.prior_mean), ")")
end

function Base.summary(io::IO, x::GaussNewtonInversion)
    print(io, "GaussNewtonInversion (prior_dim=", length(x.prior_mean), ")")
end

function Base.show(io::IO, ::MIME"text/plain", x::SparseInversion)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "SparseInversion")
        println(io, "  γ              : ", x.γ)
        println(io, "  threshold_value: ", x.threshold_value)
        println(io, "  reg            : ", x.reg)
    end
end

function Base.show(io::IO, x::SparseInversion)
    print(io, "SparseInversion (γ=", x.γ, ")")
end

function Base.summary(io::IO, x::SparseInversion)
    print(io, "SparseInversion (γ=", x.γ, ")")
end

function Base.show(io::IO, ::MIME"text/plain", x::Unscented)
    if get(io, :compact, false)
        show(io, x)
    else
        n_par = length(x.r)
        println(io, "Unscented")
        println(io, "  N_ens       : ", x.N_ens)
        println(io, "  n_params    : ", n_par)
        println(io, "  iter        : ", x.iter)
        println(io, "  α_reg       : ", x.α_reg)
        println(io, "  impose_prior: ", x.impose_prior)
    end
end

function Base.show(io::IO, x::Unscented)
    print(io, "Unscented (N_ens=", x.N_ens, ", n_params=", length(x.r), ", iter=", x.iter, ")")
end

function Base.summary(io::IO, x::Unscented)
    print(io, "Unscented (N_ens=", x.N_ens, ", n_params=", length(x.r), ", iter=", x.iter, ")")
end

function Base.show(io::IO, ::MIME"text/plain", x::TransformUnscented)
    if get(io, :compact, false)
        show(io, x)
    else
        n_par = length(x.r)
        println(io, "TransformUnscented")
        println(io, "  N_ens       : ", x.N_ens)
        println(io, "  n_params    : ", n_par)
        println(io, "  iter        : ", x.iter)
        println(io, "  α_reg       : ", x.α_reg)
        println(io, "  impose_prior: ", x.impose_prior)
    end
end

function Base.show(io::IO, x::TransformUnscented)
    print(io, "TransformUnscented (N_ens=", x.N_ens, ", n_params=", length(x.r), ", iter=", x.iter, ")")
end

function Base.summary(io::IO, x::TransformUnscented)
    print(io, "TransformUnscented (N_ens=", x.N_ens, ", n_params=", length(x.r), ", iter=", x.iter, ")")
end

# ── EnsembleKalmanProcess ─────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", x::EnsembleKalmanProcess)
    if get(io, :compact, false)
        show(io, x)
    else
        n_iter = length(x.u) - 1
        n_par = size(x.u[1].data, 1)
        println(io, "EnsembleKalmanProcess")
        println(io, "  process    : ", nameof(typeof(x.process)))
        println(io, "  N_ens      : ", x.N_ens)
        println(io, "  N_par      : ", n_par)
        println(io, "  n_iter     : ", n_iter)
        println(io, "  scheduler  : ", nameof(typeof(x.scheduler)))
        println(io, "  accelerator: ", nameof(typeof(x.accelerator)))
    end
end

function Base.show(io::IO, x::EnsembleKalmanProcess)
    n_iter = length(x.u) - 1
    print(io, "EnsembleKalmanProcess (", nameof(typeof(x.process)), ", N_ens=", x.N_ens, ", ", n_iter, " iter)")
end

function Base.summary(io::IO, x::EnsembleKalmanProcess)
    n_iter = length(x.u) - 1
    print(io, "EnsembleKalmanProcess (", nameof(typeof(x.process)), ", N_ens=", x.N_ens, ", ", n_iter, " iter)")
end
