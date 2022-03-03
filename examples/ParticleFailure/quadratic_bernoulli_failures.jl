using Distributions
using Random

"""
    quadratic_bernoulli_failures(θ::AbstractVector{FT}; p::FT = 0.1) where {FT <: Real}

Evaluates a quadratic forward model with random failures.

Inputs:
- θ   :: Parameters of the forward model.
- p   :: Probability of failure of the forward model.
"""
function quadratic_bernoulli_failures(θ::AbstractVector{FT}; p::FT = 0.1) where {FT <: Real}
    g = rand(Bernoulli(p)) ? NaN : θ' * θ
    return g
end
