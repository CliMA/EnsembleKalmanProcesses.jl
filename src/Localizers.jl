module Localizers

using Distributions
using LinearAlgebra
using DocStringExtensions
using Interpolations

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
Correlations are shrunk by a factor determined by the sample
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

Sampling error correction as of Vishny, Morzfeld, et al. (2024), [DOI](https://doi.org/10.22541/essoar.171501094.44068137/v1).
Correlations are shrunk by a factor determined by correlation and ensemble size.
The factors are automatically determined by a discrepancy principle.
Thus no algorithm parameters are required, though some tuning of the discrepancy principle tolerances are made available.

# Fields 

$(TYPEDFIELDS)

"""
struct SECNice{FT <: Real, AV <: AbstractVector} <: LocalizationMethod
    "number of samples to approximate the std of correlation distribution (default 1000)"
    n_samples::Int
    "scaling for discrepancy principle for ug correlation (default 1.0)"
    δ_ug::FT
    "scaling for discrepancy principle for gg correlation (default 1.0)"
    δ_gg::FT
    "A vector that will house a Interpolation object on first call to the localizer"
    std_of_corr::AV
end
SECNice() = SECNice(1000, 1.0, 1.0)
SECNice(δ_ug, δ_gg) = SECNice(1000, δ_ug, δ_gg)
SECNice(n_samples, δ_ug, δ_gg) = SECNice(n_samples, δ_ug, δ_gg, []) # always start with empty

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
    return kernel
end

"Uniform kernel constructor" # p::IT, d::IT, J::IT, T = Float64
function Localizer(localization::NoLocalization, J::Int, T = Float64)
    #kernel_ug = ones(T,p,d)
    return Localizer{NoLocalization, T}((cov, T, p, d, J) -> cov)
end

"Delta kernel localizer constructor"
function Localizer(localization::Delta, J::Int, T = Float64)
    #kernel_ug = T(1) * Matrix(I, p, d)
    return Localizer{Delta, T}((cov, T, p, d, J) -> kernel_function(T(1) * Matrix(I, p, d), T, p, d) .* cov)
end

function create_rbf(l, T, p, d)
    kernel_ug = zeros(T, p, d)
    for i in 1:p
        for j in 1:d
            @inbounds kernel_ug[i, j] = exp(-(i - j) * (i - j) / (2 * l * l))
        end
    end
    return kernel_ug
end
"RBF kernel localizer constructor"
function Localizer(localization::RBF, J::Int, T = Float64)
    #kernel_ug = create_rbf(localization.lengthscale,T,p,d)
    return Localizer{RBF, T}(
        (cov, T, p, d, J) -> kernel_function(create_rbf(localization_lengthscale, T, p, d), T, p, d) .* cov,
    )
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

    kernel = ones(T, p + d, p + d)
    kernel_ug = bernoulli_kernel(prob, T, p, d)
    kernel[1:p, (p + 1):end] = kernel_ug
    kernel[(p + 1):end, 1:p] = kernel_ug'
    return kernel
end

"Randomized Bernoulli dropout kernel localizer constructor"
function Localizer(localization::BernoulliDropout, J::Int, T = Float64)
    return Localizer{BernoulliDropout, T}(
        (cov, T, p, d, J) -> bernoulli_kernel_function(localization.prob, T, p, d) .* cov,
    )
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
function Localizer(localization::SEC, J::Int, T = Float64)
    return Localizer{SEC, T}((cov, T, p, d, J) -> sec(cov, localization.α, localization.r_0))
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
function Localizer(localization::SECFisher, J::Int, T = Float64)
    return Localizer{SECFisher, T}((cov, T, p, d, J) -> sec_fisher(cov, J))
end

"""
For `N_ens >= 6`: The sampling distribution of a correlation coefficient for Gaussian random variables is, under the Fisher transformation, approximately Gaussian. To estimate the standard deviation in the sampling distribution of the correlation coefficient, we draw samples from a Gaussian, apply the inverse Fisher transformation to them, and estimate an empirical standard deviation from the transformed samples.
For `N_ens < 6`: Approximate the standard deviation of correlation coefficient empirically by sampling between two correlated Gaussians of known coefficient. 
"""
function approximate_corr_std(r, N_ens, n_samples)

    if N_ens >= 6 # apply Fisher Transform
        # ρ = arctanh(r) from Fisher
        # assume r input is the mean value, i.e. assume arctanh(E(r)) = E(arctanh(r))

        ρ = r # approx solution is the identity
        #sample in ρ space
        ρ_samples = rand(Normal(0.5 * log((1 + ρ) / (1 - ρ)), 1 / sqrt(N_ens - 3)), n_samples)

        # map back through Fisher to get std of r from samples tanh(ρ)
        return std(tanh.(ρ_samples))
    else # transformation not appropriate for N < 6
        # Generate sample pairs with a correlation coefficient r
        samples_1 = rand(Normal(0, 1), N_ens, n_samples)
        samples_2 = rand(Normal(0, 1), N_ens, n_samples)
        samples_corr_with_1 = r * samples_1 + sqrt(1 - r^2) * samples_2 # will have correlation r with samples_1

        corrs = zeros(n_samples)
        for i in 1:n_samples
            corrs[i] = cor(samples_1[:, i], samples_corr_with_1[:, i])
        end
        return std(corrs)
    end

end


"""
Function that performs sampling error correction as per Vishny, Morzfeld, et al. (2024).
The input is assumed to be a covariance matrix, hence square.
"""
function sec_nice(cov, std_of_corr, δ_ug, δ_gg, N_ens, p, d)
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
        #        std_corrs = approximate_corr_std.(corr_tmp, N_ens, n_samples) # !! slowest part of code -> could speed up by precomputing/using an interpolation
        std_corrs = std_of_corr.(corr_tmp)
        std_tol = sqrt(sum(std_corrs .^ 2))
        γ_min_exceeded = max_exponent
        for γ in 2:2:max_exponent # even exponents give a PSD correction
            corr_psd = corr_tmp .^ (γ + 1) # abs not needed as γ even
            # find the first exponent that exceeds the noise tolerance in norm
            if norm(corr_psd - corr_tmp) > δ * std_tol
                γ_min_exceeded = γ
                break
            end
        end
        corr_update = corr_tmp .^ γ_min_exceeded
        corr_update_prev = corr_tmp .^ (γ_min_exceeded - 2) # previous PSD correction 

        for α in LinRange(1.0, 0.0, interp_steps)
            corr_interp = ((1 - α) * (corr_update_prev) + α * corr_update) .* corr_tmp
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


"Sampling error correction of Vishny, Morzfeld, et al. (2024) constructor"
function Localizer(localization::SECNice, J::Int, T = Float64)
    if length(localization.std_of_corr) == 0 #i.e. if the user hasn't provided an interpolation
        dr = 0.001
        grid = LinRange(-1, 1, Int(1 / dr + 1))
        std_grid = approximate_corr_std.(grid, J, localization.n_samples) # odd number to include 0           
        push!(localization.std_of_corr, linear_interpolation(grid, std_grid)) # pw-linear interpolation
    end

    return Localizer{SECNice, T}(
        (cov, T, p, d, J) -> sec_nice(cov, localization.std_of_corr[1], localization.δ_ug, localization.δ_gg, J, p, d),
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
