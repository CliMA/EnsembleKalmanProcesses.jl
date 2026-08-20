# Summary statistics H_(.) applied to a trajectory.
#
# Every statistic below acts on a trajectory matrix X of size (n_state, n_time) (rows are
# state components, columns are time samples), plus the associated time vector t_vec where
# needed, and returns one value (or one value per row of X).
using Statistics
using StatsBase
using FFTW

H_mean(X::AbstractMatrix) = vec(mean(X, dims = 2))

H_var(X::AbstractMatrix) = vec(var(X, dims = 2))

# Q_{0.05, 0.95} (or any requested set of quantiles) per state component.
function H_quantile(X::AbstractMatrix, probs = (0.05, 0.95))
    n = size(X, 1)
    Q = zeros(n, length(probs))
    for i in 1:n
        Q[i, :] = quantile(view(X, i, :), collect(probs))
    end
    return Q
end

# Leading AR(1) coefficient (lag-1 autocorrelation), per state component.
function H_ac(X::AbstractMatrix)
    n = size(X, 1)
    λ = zeros(n)
    for i in 1:n
        λ[i] = autocor(view(X, i, :), 1:1)[1]
    end
    return λ
end

# Full autocorrelation function up to `maxlag`, per state component (used only for
# diagnostics/plotting, H_ac above is the single-number summary statistic).
function autocorrelation_function(X::AbstractMatrix, maxlag::Int)
    n = size(X, 1)
    acf = zeros(n, maxlag + 1)
    for i in 1:n
        acf[i, :] = autocor(view(X, i, :), 0:maxlag)
    end
    return acf
end

# H_spec(X, [ω_min, ω_max]) = (1/T) ∫_{ω_min}^{ω_max} x̂_T(ω) dω,
# x̂_T(ω) = ∫_0^T (x(t) - mean(x)) e^{-iωt} dt,
# approximated here via the (uniformly-sampled) discrete Fourier transform. Returns one
# complex number per state component (row of X). Requires uniformly-sampled t_vec.
function H_spec(X::AbstractMatrix, t_vec::AbstractVector, ω_band::Tuple{<:Real, <:Real})
    n, N = size(X)
    dt = t_vec[2] - t_vec[1]
    T = t_vec[end] - t_vec[1]
    freqs = 2π .* FFTW.fftfreq(N, 1 / dt) # angular frequency grid matching fft(...) ordering
    dω = freqs[2] - freqs[1]
    ω_min, ω_max = ω_band
    band_idx = findall(f -> ω_min <= f <= ω_max, freqs)
    out = zeros(ComplexF64, n)
    for i in 1:n
        xi = X[i, :] .- mean(view(X, i, :))
        x̂ = fft(xi) .* dt
        out[i] = sum(view(x̂, band_idx)) * dω / T
    end
    return out
end

# Power spectral density |x̂_T(ω)|^2 / T on the (non-negative, ordered) angular-frequency
# grid, per state component. Diagnostic-only (not one of the H_(.) calibration statistics),
# used to visually locate the characteristic timescales of the multiscale system.
function power_spectrum(X::AbstractMatrix, t_vec::AbstractVector)
    n, N = size(X)
    dt = t_vec[2] - t_vec[1]
    T = t_vec[end] - t_vec[1]
    freqs = 2π .* FFTW.fftfreq(N, 1 / dt)
    order = sortperm(freqs)
    psd = zeros(n, N)
    for i in 1:n
        xi = X[i, :] .- mean(view(X, i, :))
        x̂ = fft(xi) .* dt
        psd[i, :] = (abs.(x̂) .^ 2) ./ T
    end
    return freqs[order], psd[:, order]
end
