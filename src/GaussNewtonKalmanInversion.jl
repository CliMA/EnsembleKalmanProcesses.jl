#Gauss Newton Kalman Inversion: specific structures and function definitions

"""
    GaussNewtonInversion <: Process

A Gauss Newton Kalman Inversion process
"""
struct GaussNewtonInversion{VV <: AbstractVector, AMorUS <: Union{AbstractMatrix, UniformScaling}} <: Process 
    "Mean of Gaussian parameter prior in unconstrained space"
    prior_mean::VV
    "Covariance of Gaussian parameter prior in unconstrained space"
    prior_cov::AMorUS
end

function GaussNewtonInversion(prior::ParameterDistribution)
    mean_prior = Vector(mean(prior))
    cov_prior = Matrix(cov(prior))
    FT = eltype(mean_prior)
    return GaussNewtonInversion{FT}(mean_prior, cov_prior)
end


                            
