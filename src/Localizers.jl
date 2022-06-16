module Localizers

using Distributions
using LinearAlgebra
using DocStringExtensions

export NoLocalization, Delta, RBF, BernoulliDropout, SEC, SECFisher
export LocalizationMethod, Localizer

abstract type LocalizationMethod end

struct NoLocalization <: LocalizationMethod end
struct Delta <: LocalizationMethod end

"""
    RBF{FT <: Real} <: LocalizationMethod

Radial basis function localization method. Covariance
terms are damped as d(i,j)= |i-j|/l increases, following
a Gaussian.

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
probability 1-p, retaining a Hermitian structure.

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
factor of |r|^α, as per Lee (2021).

Lee, Y. (2021). Sampling error correction in ensemble
Kalman inversion. arXiv:2105.11341 [cs, math].
http://arxiv.org/abs/2105.11341


# Fields

$(TYPEDFIELDS)
"""
struct SEC{FT <: Real} <: LocalizationMethod
    "Controls degree of sampling error correction"
    α::FT
end

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

struct Localizer{LM <: LocalizationMethod, T}
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
    return return Localizer{NoLocalization, T}(kernel_function(kernel_ug, T, p, d))
end

"Delta kernel localizer constructor"
function Localizer(localization::Delta, p::IT, d::IT, J::IT, T = Float64) where {IT <: Int}
    kernel_ug = T(1) * Matrix(I, p, d)
    return return Localizer{Delta, T}(kernel_function(kernel_ug, T, p, d))
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
    return return Localizer{RBF, T}(kernel_function(kernel_ug, T, p, d))
end

"Randomized Bernoulli dropout kernel localizer constructor"
function Localizer(localization::BernoulliDropout, p::IT, d::IT, J::IT, T = Float64) where {IT <: Int}
    kernel_ug = T(1) * Matrix(I, p, d)

    # Transpose
    kernel_ug = p < d ? kernel_ug' : kernel_ug
    for i in 2:size(kernel_ug, 1)
        for j in 1:min(i - 1, size(kernel_ug, 2))
            @inbounds kernel_ug[i, j] = T(rand(Bernoulli(localization.prob)))
            if i <= size(kernel_ug, 2)
                kernel_ug[j, i] = kernel_ug[i, j]
            end
        end
    end
    # Transpose back
    kernel_ug = p < d ? kernel_ug' : kernel_ug

    return return Localizer{BernoulliDropout, T}(kernel_function(kernel_ug, T, p, d))
end

"""
Function that performs sampling error correction as per
Lee (2021).
"""
function sec(cov, α)
    v = sqrt.(diag(cov))
    V = Diagonal(v)
    V_inv = inv(V)
    R = V_inv * cov * V_inv
    R_sec = R .* (abs.(R) .^ α)
    return V * R_sec * V
end

"Sampling error correction (Lee, 2021) constructor"
function Localizer(localization::SEC, p::IT, d::IT, J::IT, T = Float64) where {IT <: Int}
    return Localizer{SEC, T}((cov) -> sec(cov, localization.alpha))
end

"""
Function that performs sampling error correction as per
Flowerdew (2015).
"""
function sec_fisher(cov, N_ens)
    # Decompose covariance matrix C = V*R*V, where R is the
    # correlation matrix and V is a diagonal matrix holding the
    # standard deviations.
    v = sqrt.(diag(cov))
    V = Diagonal(v)
    V_inv = inv(V)
    R = V_inv * cov * V_inv

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

end # module
