module Localizers

using Distributions
using LinearAlgebra
using DocStringExtensions

export NoLocalization, Delta, RBF, BernoulliDropout, SEC, SECFisher, LWShrinkage
export LocalizationMethod, Localizer

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

struct LWShrinkage <: LocalizationMethod end

function lw_shrinkage(sample_mat::AA) where {AA <: AbstractMatrix}
    n_out, n_sample = size(sample_mat)

    # de-mean (as we will use the samples directly for calculation of β)
    sample_mat_zeromean = sample_mat .- mean(sample_mat, dims=2)
    # Ledoit Wolf shrinkage to I

    # get sample covariance
    Γ = cov(sample_mat_zeromean, dims = 2)
    # estimate opt shrinkage
    μ_shrink = 1/n_out * tr(Γ)
    δ_shrink = norm(Γ - μ_shrink*I)^2 / n_out # (scaled) frob norm of Γ_m
    #once de-meaning, we need to correct the sample covariance with an n_sample -> n_sample-1
    β_shrink = sum([norm(c*c'-   - Γ)^2/n_out for c in eachcol(sample_mat_zeromean)])/ (n_sample-1)^2 

    γ_shrink = min(β_shrink / δ_shrink, 1) # clipping is typically rare
    #  γμI + (1-γ)Γ
    Γ .*= (1-γ_shrink)
    for i = 1:n_out
        Γ[i,i] += γ_shrink * μ_shrink 
    end 

    @info "Shrinkage scale: $(γ_shrink), (0 = none, 1 = revert to scaled Identity)\n shrinkage covariance condition number: $(cond(Γ))"
    return Γ
end        

"ShrinkageEstimator constructor"
function Localizer(localization::LWShrinkage, p::IT, d::IT, J::IT, T = Float64) where {IT <: Int}
    return Localizer{LWShrinkage, T}(samples -> lw_shrinkage(samples))
end

"""
    get_localizer(loc::Localizer)
Return localizer type.
"""
get_localizer(loc::Localizer{T1,T2}) where {T1, T2} = T1



end # module
