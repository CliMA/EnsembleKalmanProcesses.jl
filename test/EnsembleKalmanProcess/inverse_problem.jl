using Distributions
using LinearAlgebra
using Random

using EnsembleKalmanProcesses.ParameterDistributions

"""
    G_linear(ϕ, A)

Linear forward map returning Aϕ
"""
function G_linear(ϕ, A)
    return A * ϕ
end

"""
    linear_map(rng, n_obs)

Randomly chosen linear map described by a MvNormal operator with nonzero correlation
"""
function linear_map(rng, n_obs)
    cross_corr = rand(rng, Uniform(-0.9, 0.9))
    C = [1 -cross_corr; -cross_corr 1]          # Correlation structure for linear operator
    return rand(rng, MvNormal(zeros(size(C, 1)), C), n_obs)'    # Linear operator in R^{n_obs x 2}
end

"""
    G_nonlinear(ϕ, rng, noise_dist, ϕ_star = 0.0)

Noisy nonlinear map output given `ϕ` and multivariate noise.
The map: √(sum(ϕ - ϕ_star)²)

Inputs:

 - `ϕ`                           :: Input to the forward map (constrained space)
 - `rng`                         :: Random number generator
 - `noise_dist`                  :: Multivariate noise distribution from which stochasticity of map is sampled.
 - `ϕ_star`                      :: Minimum of the function
 - `add_or_mult_noise = nothing` :: additive ["add"] or multiplicative ["mult"] variability optional

"""
function G_nonlinear(ϕ, rng, noise_dist; ϕ_star = 0.0, add_or_mult_noise = nothing)
    if ndims(ϕ) == 1
        p = length(ϕ)
        ϕ_new = reshape(ϕ, (p, 1))
    else
        ϕ_new = ϕ
    end
    true_output = sqrt.(sum((ϕ_new .- ϕ_star) .^ 2, dims = 1)) # (1 x n_ens)
    # Add noise from noise_dist

    if isnothing(add_or_mult_noise)
        #just copy the observation n_obs times
        output = ones(length(noise_dist), 1) * true_output # (n_obs x n_ens)
    elseif add_or_mult_noise == "add"
        output = rand(rng, noise_dist, size(true_output, 2)) .+ true_output # (n_obs x n_ens)   
    elseif add_or_mult_noise == "mult"
        output = (1 .+ rand(rng, noise_dist)) * true_output # (n_obs x n_ens)
    else
        throw(
            ArgumentError(
                "`add_or_mult_noise` keyword expects either nothing, \"add\" or \"mult\". Got $(add_or_mult_noise).",
            ),
        )
    end
    return size(output, 2) == 1 ? vec(output) : output
end

"""
    linear_inv_problem(ϕ_star, noise_level, n_obs, rng; obs_corrmat = I, return_matrix = false)

Returns the objects defining a linear inverse problem.

Inputs:

 - `ϕ_star`          :: True parameter value (constrained space)
 - `noise_level`     :: Magnitude of the observational noise
 - `n_obs`           :: Dimension of the observational vector
 - `rng`             :: Random number generator
 - `obs_corrmat`     :: Correlation structure of the observational noise
 - `return_matrix`   :: If true, returns the forward map operator

Returns:

 - `y_obs`           :: Vector of observations
 - `G`               :: Map from constrained space to data space
 - `Γ`               :: Observational covariance matrix
 - `A`               :: Forward model operator from constrained space
"""
function linear_inv_problem(ϕ_star, noise_level, n_obs, rng; obs_corrmat = I, return_matrix = false)
    A = linear_map(rng, n_obs)
    # Define forward map from unconstrained space directly
    G(x) = G_linear(x, A)
    y_star = G(ϕ_star)
    Γ = noise_level^2 * obs_corrmat
    noise = MvNormal(zeros(n_obs), Γ)
    y_obs = y_star .+ rand(rng, noise)
    if return_matrix
        return y_obs, G, Γ, A
    else
        return y_obs, G, Γ
    end
end

"""
    nonlinear_inv_problem(ϕ_star, noise_level, n_obs, rng; obs_corrmat = I; add_or_mult_noise = "add")

Returns the objects defining a random nonlinear inverse problem

Inputs:

 - `ϕ_star`                      :: True parameter value (constrained space)
 - `noise_level`                 :: Magnitude of the observational noise
 - `n_obs`                       :: Dimension of the observational vector
 - `rng`                         :: Random number generator
 - `obs_corrmat = I  `           :: Correlation structure of the observational noise 
 - `add_or_mult_noise = nothing` :: additive ["add"] or multiplicative ["mult"] variability optional

Returns:

 - `y_obs`           :: Vector of observations
 - `G`               :: Map from constrained space to data space
 - `Γ`               :: Observational covariance matrix
"""

function nonlinear_inv_problem(ϕ_star, noise_level, n_obs, rng; obs_corrmat = I, add_or_mult_noise = nothing)
    Γ = noise_level^2 * obs_corrmat
    noise = MvNormal(zeros(n_obs), Γ)
    # Define forward map from unconstrained space directly
    G(x) = G_nonlinear(x, rng, noise, ϕ_star = ϕ_star, add_or_mult_noise = add_or_mult_noise)
    # Get observation
    y_star = G(ϕ_star)
    y_obs = y_star .+ rand(rng, noise)
    return y_obs, G, Γ
end
