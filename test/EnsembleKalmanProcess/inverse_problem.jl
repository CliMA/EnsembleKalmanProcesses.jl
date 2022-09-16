using Distributions
using LinearAlgebra
using Random

using EnsembleKalmanProcesses.ParameterDistributions

"Linear forward map output"
function G_linear(ϕ, A)
    return A * ϕ
end

"Randomized linear map from R^2 to n_obs"
function linear_map(rng, n_obs)
    cross_corr = rand(Uniform(-0.9, 0.9))
    C = [1 -cross_corr; -cross_corr 1]          # Correlation structure for linear operator
    return rand(rng, MvNormal(zeros(size(C, 1)), C), n_obs)'    # Linear operator in R^{n_obs x 2}
end

"""
    G_nonlinear(ϕ, rng, noise_dist, ϕ_star = 0.0)

Noisy nonlinear map output given `ϕ` and multivariate noise.

Inputs:

 - `ϕ`             :: Input to the forward map (constrained space)
 - `rng`           :: Random number generator
 - `noise_dist`    :: Multivariate noise distribution from which stochasticity of map is sampled.
 - `ϕ_star`        :: Minimum of the function
"""
function G_nonlinear(ϕ, rng, noise_dist, ϕ_star = 0.0)
    if ndims(ϕ) == 1
        p = length(ϕ)
        ϕ_new = reshape(ϕ, (p, 1))
    else
        ϕ_new = ϕ
    end
    true_output = sqrt.(sum((ϕ_new .- ϕ_star) .^ 2, dims = 1)) # (1 x n_ens)
    # Add noise from noise_dist
    output = (1 .+ rand(rng, noise_dist)) * true_output # (n_obs x n_ens)
    return size(output, 2) == 1 ? vec(output) : output
end

"""
    linear_inv_problem(ϕ_star, noise_level, n_obs, prior, rng; obs_corrmat = I, return_matrix = false)

Returns the objects defining a linear inverse problem.

Inputs:

 - `ϕ_star`          :: True parameter value (constrained space)
 - `noise_level`     :: Magnitude of the observational noise
 - `n_obs`           :: Dimension of the observational vector
 - `prior`           :: Prior parameter distribution
 - `rng`             :: Random number generator
 - `obs_corrmat`     :: Correlation structure of the observational noise
 - `return_matrix`   :: If true, returns the forward map operator

Returns:

 - `y_obs`           :: Vector of observations
 - `G`               :: Map from unconstrained space to data space
 - `Γ`               :: Observational covariance matrix
 - `A`               :: Forward model operator from constrained space
"""
function linear_inv_problem(ϕ_star, noise_level, n_obs, prior, rng; obs_corrmat = I, return_matrix = false)
    A = linear_map(rng, n_obs)
    # Define forward map from unconstrained space directly
    G(x) = G_linear(transform_unconstrained_to_constrained(prior, x), A)
    y_star = G_linear(ϕ_star, A)
    Γ = noise_level^2 * obs_corrmat
    noise = MvNormal(zeros(n_obs), Γ)
    y_obs = y_star .+ rand(rng, noise)
    if return_matrix
        return y_obs, G, Γ, A
    else
        return y_obs, G, Γ
    end
end

"Returns the objects defining a nonlinear inverse problem"
function nonlinear_inv_problem(ϕ_star, noise_level, n_obs, prior, rng; obs_corrmat = I)
    Γ = noise_level^2 * obs_corrmat
    noise = MvNormal(zeros(n_obs), Γ)
    # Define forward map from unconstrained space directly
    G(x) = G_nonlinear(transform_unconstrained_to_constrained(prior, x), rng, noise, ϕ_star)
    # Get observation
    y_star = G_nonlinear(ϕ_star, rng, noise, ϕ_star)
    y_obs = y_star .+ rand(rng, noise)
    return y_obs, G, Γ
end
