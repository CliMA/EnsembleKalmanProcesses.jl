module Localizers

using Distributions
using LinearAlgebra
using DocStringExtensions

export NoLocalization, Delta, RBF, BernoulliDropout
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

struct Localizer{LM <: LocalizationMethod, T}
    kernel::Union{AbstractMatrix{T}, UniformScaling{T}}
end

"Uniform kernel constructor"
function Localizer(localization::NoLocalization, p::IT, d::IT, T = Float64) where {IT <: Int}
    kernel = ones(T, p, d)
    return Localizer{NoLocalization, T}(kernel)
end

"Delta kernel localizer constructor"
function Localizer(localization::Delta, p::IT, d::IT, T = Float64) where {IT <: Int}
    kernel = T(1) * Matrix(I, p, d)
    return Localizer{Delta, T}(kernel)
end

"RBF kernel localizer constructor"
function Localizer(localization::RBF, p::IT, d::IT, T = Float64) where {IT <: Int}
    l = localization.lengthscale
    kernel = zeros(T, p, d)
    for i in 1:p
        for j in 1:d
            @inbounds kernel[i, j] = exp(-(i - j) * (i - j) / (2 * l * l))
        end
    end
    return Localizer{RBF, T}(kernel)
end

"Randomized Bernoulli dropout kernel localizer constructor"
function Localizer(localization::BernoulliDropout, p::IT, d::IT, T = Float64) where {IT <: Int}
    kernel = T(1) * Matrix(I, p, d)

    # Transpose
    kernel = p < d ? kernel' : kernel
    for i in 2:size(kernel, 1)
        for j in 1:min(i - 1, size(kernel, 2))
            @inbounds kernel[i, j] = T(rand(Bernoulli(localization.prob)))
            if i <= size(kernel, 2)
                kernel[j, i] = kernel[i, j]
            end
        end
    end
    # Transpose back
    kernel = p < d ? kernel' : kernel

    return Localizer{BernoulliDropout, T}(kernel)
end

end # module
