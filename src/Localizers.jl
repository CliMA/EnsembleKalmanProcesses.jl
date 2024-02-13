module Localizers

using Distributions
using LinearAlgebra
using DocStringExtensions

export NoLocalization, Delta, RBF, BernoulliDropout, SEC, SECFisher, SECNice
export LocalizationMethod, Localizer
export approximate_corr_std
abstract type LocalizationMethod end

"Idempotent localization method."
struct NoLocalization <: LocalizationMethod end

"Dirac delta localization method, with an identity matrix as the kernel."
struct Delta <: LocalizationMethod end

"""
    RBF{FT <: Real} <: LocalizationMethod

Radial basis function localization method. Covariance
terms ``C_{i,j}`` are damped through multiplication with a
centered Gaussian with standardized deviation ``d(i,j)= \\vert i-j \\vert / l``.

# Fields

$(TYPEDFIELDS)
"""
struct RBF{FT <: Real} <: LocalizationMethod
    "Length scale defining the RBF kernel"
    lengthscale::FT
end

"""
    BernoulliDropout{FT <: Real} <: LocalizationMethod

Localization method that drops cross-covariance terms with
probability ``1-p``, retaining a Hermitian structure.

# Fields

$(TYPEDFIELDS)
"""
struct BernoulliDropout{FT <: Real} <: LocalizationMethod
    "Probability of keeping a given cross-covariance term"
    prob::FT
end

"""
    SEC{FT <: Real} <: LocalizationMethod

Sampling error correction that shrinks correlations by a
factor of ``\\vert r \\vert ^\\alpha``, as per Lee (2021). Sparsity of the
resulting correlations can be imposed through the parameter
`r_0`.

Lee, Y. (2021). Sampling error correction in ensemble
Kalman inversion. arXiv:2105.11341 [cs, math].
http://arxiv.org/abs/2105.11341


# Fields

$(TYPEDFIELDS)
"""
struct SEC{FT <: Real} <: LocalizationMethod
    "Controls degree of sampling error correction"
    α::FT
    "Cutoff correlation"
    r_0::FT
end
SEC(α) = SEC{eltype(α)}(α, eltype(α)(0))

"""
    SECFisher <: LocalizationMethod

Sampling error correction for EKI, as per Lee (2021), but using
the method from Flowerdew (2015) based on the Fisher transformation.
Correlations are shrinked by a factor determined by the sample
correlation and the ensemble size. 

Flowerdew, J. (2015). Towards a theory of optimal localisation.
Tellus A: Dynamic Meteorology and Oceanography, 67(1), 25257.
https://doi.org/10.3402/tellusa.v67.25257

Lee, Y. (2021). Sampling error correction in ensemble
Kalman inversion. arXiv:2105.11341 [cs, math].
http://arxiv.org/abs/2105.11341
"""
struct SECFisher <: LocalizationMethod end


"""
    SECNice{FT <: Real} <: LocalizationMethod

Sample error correction as of Morzfeld, Vishny et al. (2024).
Correlations are shrinked by a factor determined by correlation and ensemble size.
The factors are automatically determined by a discrepancy principle.
Thus no algorithm parameters are required, though some tuning of the discrepancy principle tolerances are made available.

# Fields 

$(TYPEDFIELDS)

"""
struct SECNice{FT <: Real} <: LocalizationMethod
    "number of samples to approximate the std of correlation distribution (default 1000)"
    n_samples::Int
    "scaling for discrepancy principle for ug correlation (default 1.0)"
    tol_ug::FT
    "scaling for discrepancy principle for gg correlation (default 1.0)"
    tol_gg::FT
end
SECNice() = SECNice(1000, 1.0, 1.0)
SECNice(n_samples) = SECNice(n_samples, 1.0, 1.0)
"""
    Localizer{LM <: LocalizationMethod, T}

Structure that defines a `localize` function, based on
a localization method.

# Fields

$(TYPEDFIELDS)

# Constructors

$(METHODLIST)
"""
struct Localizer{LM <: LocalizationMethod, T}
    "Localizing function of the form: `cov -> kernel .* cov`"
    localize::Function
end

"""
Localize using a Schur product with a kernel matrix. Only the u–G(u) block is localized.
"""
function kernel_function(kernel_ug, T, p, d)
    kernel = ones(T, p + d, p + d)
    kernel[1:p, (p + 1):end] = kernel_ug
    kernel[(p + 1):end, 1:p] = kernel_ug'
    return (cov) -> kernel .* cov
end

"Uniform kernel constructor"
function Localizer(localization::NoLocalization, p::IT, d::IT, J::IT, T = Float64) where {IT <: Int}
    kernel_ug = ones(T, p, d)
    return Localizer{NoLocalization, T}(kernel_function(kernel_ug, T, p, d))
end

"Delta kernel localizer constructor"
function Localizer(localization::Delta, p::IT, d::IT, J::IT, T = Float64) where {IT <: Int}
    kernel_ug = T(1) * Matrix(I, p, d)
    return Localizer{Delta, T}(kernel_function(kernel_ug, T, p, d))
end

"RBF kernel localizer constructor"
function Localizer(localization::RBF, p::IT, d::IT, J::IT, T = Float64) where {IT <: Int}
    l = localization.lengthscale
    kernel_ug = zeros(T, p, d)
    for i in 1:p
        for j in 1:d
            @inbounds kernel_ug[i, j] = exp(-(i - j) * (i - j) / (2 * l * l))
        end
    end
    return Localizer{RBF, T}(kernel_function(kernel_ug, T, p, d))
end

"Localization kernel with Bernoulli trials as off-diagonal terms (symmetric)"
function bernoulli_kernel(prob, T, p, d)
    kernel_ug = T(1) * Matrix(I, p, d)
    # Transpose
    kernel_ug = p < d ? kernel_ug' : kernel_ug
    # Add correlations as Bernoulli events
    for i in 2:size(kernel_ug, 1)
        for j in 1:min(i - 1, size(kernel_ug, 2))
            @inbounds kernel_ug[i, j] = T(rand(Bernoulli(prob)))
            if i <= size(kernel_ug, 2)
                kernel_ug[j, i] = kernel_ug[i, j]
            end
        end
    end
    # Transpose back
    kernel_ug = p < d ? kernel_ug' : kernel_ug
    return kernel_ug
end

"""
Localize using a Schur product with a random draw of a Bernoulli kernel matrix. Only the u–G(u) block is localized.
"""
function bernoulli_kernel_function(prob, T, p, d)
    function get_kernel()
        kernel = ones(T, p + d, p + d)
        kernel_ug = bernoulli_kernel(prob, T, p, d)
        kernel[1:p, (p + 1):end] = kernel_ug
        kernel[(p + 1):end, 1:p] = kernel_ug'
        return kernel
    end
    return (cov) -> get_kernel() .* cov
end

"Randomized Bernoulli dropout kernel localizer constructor"
function Localizer(localization::BernoulliDropout, p::IT, d::IT, J::IT, T = Float64) where {IT <: Int}
    return Localizer{BernoulliDropout, T}(bernoulli_kernel_function(localization.prob, T, p, d))
end

"""
Function that performs sampling error correction as per Lee (2021).
The input is assumed to be a covariance matrix, hence square.
"""
function sec(cov, α, r_0)
    v = sqrt.(diag(cov))
    V = Diagonal(v)
    V_inv = inv(V)
    R = V_inv * cov * V_inv
    R_sec = R .* (abs.(R) .^ α)
    # Apply cutoff
    R_sec = R_sec .* (abs.(R_sec) .> r_0)
    return V * R_sec * V
end

"Sampling error correction (Lee, 2021) constructor"
function Localizer(localization::SEC, p::IT, d::IT, J::IT, T = Float64) where {IT <: Int}
    return Localizer{SEC, T}((cov) -> sec(cov, localization.α, localization.r_0))
end

"""
Function that performs sampling error correction as per Flowerdew (2015).
The input is assumed to be a covariance matrix, hence square.
"""
function sec_fisher(cov, N_ens)
    # Decompose covariance matrix C = V*R*V, where R is the
    # correlation matrix and V is a diagonal matrix holding the
    # standard deviations.
    v = sqrt.(diag(cov))
    V = Diagonal(v)
    V_inv = inv(V)
    R = V_inv * cov * V_inv
    bd_tol = 1e8 * eps()
    clamp!(R, -1 + bd_tol, 1 - bd_tol)
    R_sec = zeros(size(R))
    for i in 1:size(R)[1]
        for j in 1:i
            r = R[i, j]
            if (i == j) | (r >= 1)
                R_sec[i, j] = R_sec[j, i] = 1
            else
                # Apply Fisher transformation
                s = atanh(r)
                σ_s = 1 / sqrt(N_ens - 3)
                # Estimate σ_r as half of (s + σ_s) - (s - σ_s), transformed back into r space
                σ_r = (tanh(s + σ_s) - tanh(s - σ_s)) / 2
                Q = r / σ_r
                alpha = (Q^2) / (1 + Q^2)
                R_sec[i, j] = R_sec[j, i] = alpha * r
            end
        end
    end
    return V * R_sec * V
end

"Sampling error correction (Flowerdew, 2015) constructor"
function Localizer(localization::SECFisher, p::IT, d::IT, J::IT, T = Float64) where {IT <: Int}
    return Localizer{SECFisher, T}((cov) -> sec_fisher(cov, J))
end

function approximate_corr_std(r, N_ens, n_samples)
    # ρ = arctanh(r) from Fisher
    # assume r input is the mean value, i.e. assume arctanh(E(r)) = E(arctanh(r))

    ρ = r # approx solution is the identity
    #sample in ρ space
    ρ_samples = rand(Normal(0.5 * log((1 + ρ) / (1 - ρ)), 1 / sqrt(N_ens - 3)), n_samples) # N_ens

    # map back through Fisher to get std of r from samples tanh(ρ)
    return std(tanh.(ρ_samples))
end


"""
Function that performs sampling error correction as per Morzfeld, Vishny (2024).
The input is assumed to be a covariance matrix, hence square.
"""
function sec_nice(cov, n_samples, δ_ug, δ_gg, N_ens, p, d)
    if N_ens < 6
        @warn "significant localization approximation error may occur for ensemble size below 6. Here, ensemble size = $N_ens"
    end
    bd_tol = 1e8 * eps()

    v = sqrt.(diag(cov))
    V = Diagonal(v) #stds
    V_inv = inv(V)
    corr = clamp.(V_inv * cov * V_inv, -1 + bd_tol, 1 - bd_tol) # full corr
    # parameter sweep over the exponents
    max_exponent = 2 * 5 # must be even
    interp_steps = 10

    ug_idx = [1:p, (p + 1):(p + d)]
    ugt_idx = [(p + 1):(p + d), 1:p] # don't loop over this one
    gg_idx = [(p + 1):(p + d), (p + 1):(p + d)]

    corr_updated = copy(corr)
    for (idx_set, δ) in zip([ug_idx, gg_idx], [δ_ug, δ_gg])

        corr_tmp = corr[idx_set...]
        # use find the variability in the corr coeff matrix entries
        std_corrs = approximate_corr_std.(corr_tmp, N_ens, n_samples) # !! slowest part of code -> could speed up by precomputing/using an interpolation
        std_tol = sqrt(sum(std_corrs .^ 2))
        α_min_exceeded = [max_exponent]
        for α in 2:2:max_exponent # even exponents give a PSD correction
            corr_psd = corr_tmp .^ (α + 1) # abs not needed as α even
            # find the first exponent that exceeds the noise tolerance in norm
            if norm(corr_psd - corr_tmp) > δ * std_tol
                α_min_exceeded[1] = α
                break
            end
        end
        corr_psd = corr_tmp .^ α_min_exceeded[1]
        corr_psd_prev = corr_tmp .^ (α_min_exceeded[1] - 2) # previous PSD correction 

        for α in LinRange(1.0, 0.0, interp_steps)
            corr_interp = ((1 - α) * (corr_psd_prev) + α * corr_psd) .* corr_tmp
            if norm(corr_interp - corr_tmp) < δ * std_tol
                corr_updated[idx_set...] = corr_interp #update the correlation matrix block
                break
            end
        end

    end

    # finally correct the ug'
    corr_updated[ugt_idx...] = corr_updated[ug_idx...]'

    return V * corr_updated * V # rebuild the cov matrix

end


"Sampling error correction (Morzfeld, Vishny et al., 2024) constructor"
function Localizer(localization::SECNice, p::IT, d::IT, J::IT, T = Float64) where {IT <: Int}
    return Localizer{SECNice, T}(
        (cov) -> sec_nice(cov, localization.n_samples, localization.tol_ug, localization.tol_gg, J, p, d),
    )
end





# utilities
"""
    get_localizer(loc::Localizer)
Return localizer type.
"""
function get_localizer(loc::Localizer{T1, T2}) where {T1, T2}
    return T1
end







end # module
