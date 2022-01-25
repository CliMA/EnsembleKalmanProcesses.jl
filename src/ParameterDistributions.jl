module ParameterDistributions

## Usings
using Distributions
using Statistics
using Random

#import (to add definitions)
import StatsBase: mean, var, cov,sample

## Exports

#types
export ParameterDistributionType

#objects
export Parameterized, Samples
export ParameterDistribution
export Constraint

#functions
export get_name, get_distribution, get_total_dimension, get_dimensions, get_all_constraints, get_n_samples
export sample_distribution
export no_constraint, bounded_below, bounded_above, bounded
export transform_constrained_to_unconstrained, transform_unconstrained_to_constrained
export get_logpdf, batch
#export get_cov, get_var, get_mean
## Objects
# for the Distribution
abstract type ParameterDistributionType end

"""
    Parameterized <: ParameterDistributionType
    
A distribution constructed from a parametrized formula (e.g Julia Distributions.jl)
"""
struct Parameterized <: ParameterDistributionType
    distribution::Distribution
end

"""
    Samples{FT <: Real} <: ParameterDistributionType

A distribution comprised of only samples, stored as columns of parameters.
"""
struct Samples{FT <: Real} <: ParameterDistributionType
    distribution_samples::Array{FT, 2} #parameters are columns
    Samples(distribution_samples::Array{FT, 2}; params_are_columns = true) where {FT <: Real} =
        params_are_columns ? new{FT}(distribution_samples) : new{FT}(permutedims(distribution_samples, (2, 1)))
    #Distinguish 1 sample of an ND parameter or N samples of 1D parameter, and store as 2D array  
    Samples(distribution_samples::Array{FT, 1}; params_are_columns = true) where {FT <: Real} =
        params_are_columns ? new{FT}(reshape(distribution_samples, 1, :)) : new{FT}(reshape(distribution_samples, :, 1))
end


# For the transforms
abstract type ConstraintType end

"""
    Constraint <: ConstraintType

Contains two functions to map between constrained and unconstrained spaces.
"""
struct Constraint <: ConstraintType
    constrained_to_unconstrained::Function
    unconstrained_to_constrained::Function
end


"""
    no_constraint()

Constructs a Constraint with no constraints, enforced by maps x -> x and x -> x.
"""
function no_constraint()
    c_to_u = (x -> x)
    u_to_c = (x -> x)
    return Constraint(c_to_u, u_to_c)
end

"""
    bounded_below(lower_bound::FT) where {FT <: Real}

Constructs a Constraint with provided lower bound, enforced by maps `x -> log(x - lower_bound)`
and `x -> exp(x) + lower_bound`.
"""
function bounded_below(lower_bound::FT) where {FT <: Real}
    c_to_u = (x -> log(x - lower_bound))
    u_to_c = (x -> exp(x) + lower_bound)
    return Constraint(c_to_u, u_to_c)
end

"""
    bounded_above(upper_bound::FT) where {FT <: Real} 

Constructs a Constraint with provided upper bound, enforced by maps `x -> log(upper_bound - x)`
and `x -> upper_bound - exp(x)`.
"""
function bounded_above(upper_bound::FT) where {FT <: Real}
    c_to_u = (x -> log(upper_bound - x))
    u_to_c = (x -> upper_bound - exp(x))
    return Constraint(c_to_u, u_to_c)
end


"""
    bounded(lower_bound::FT, upper_bound::FT) where {FT <: Real} 

Constructs a Constraint with provided upper and lower bounds, enforced by maps
`x -> log((x - lower_bound) / (upper_bound - x))`
and `x -> (upper_bound * exp(x) + lower_bound) / (exp(x) + 1)`.

"""
function bounded(lower_bound::FT, upper_bound::FT) where {FT <: Real}
    if (upper_bound <= lower_bound)
        throw(DomainError("upper bound must be greater than lower bound"))
    end
    c_to_u = (x -> log((x - lower_bound) / (upper_bound - x)))
    u_to_c = (x -> (upper_bound * exp(x) + lower_bound) / (exp(x) + 1))
    return Constraint(c_to_u, u_to_c)
end

"""
    len(c::Array{CType})

The number of constraints, each constraint has length 1.
"""
len(c::CType) where {CType <: ConstraintType} = 1

len(carray::Array{CType}) where {CType <: ConstraintType} = size(carray)[1]

"""
    get_dimensions(d<:ParametrizedDistributionType)

The number of dimensions of the parameter space
"""
dimension(d::Parameterized) = length(d.distribution)

dimension(d::Samples) = size(d.distribution_samples)[1]

"""
    n_samples(d::Samples)

The number of samples in the array.
"""
n_samples(d::Samples) = size(d.distribution_samples)[2]

n_samples(d::Parameterized) =
    "Distribution stored in Parameterized form, draw samples using `sample_distribution` function"

"""
    ParameterDistribution

Structure to hold a parameter distribution, always stored as an array of distributions.
"""
struct ParameterDistribution{PDType <: ParameterDistributionType, CType <: ConstraintType, ST <: AbstractString}
    distributions::Array{PDType}
    constraints::Array{CType}
    names::Array{ST}

    function ParameterDistribution(
        parameter_distributions::Union{PDType, Array{PDType}},
        constraints::Union{CType, Array{CType}, Array},
        names::Union{ST, Array{ST}},
    ) where {PDType <: ParameterDistributionType, CType <: ConstraintType, ST <: AbstractString}

        parameter_distributions =
            isa(parameter_distributions, PDType) ? [parameter_distributions] : parameter_distributions
        n_parameters_per_dist = [dimension(pd) for pd in parameter_distributions]
        constraints = isa(constraints, Union{<:ConstraintType, Array{<:ConstraintType}}) ? [constraints] : constraints #to calc n_constraints_per_dist
        names = isa(names, ST) ? [names] : names

        n_constraints_per_dist = [len(c) for c in constraints]
        n_dists = length(parameter_distributions)
        n_names = length(names)
        if !(n_parameters_per_dist == n_constraints_per_dist)
            throw(DimensionMismatch("There must be one constraint per parameter in a distribution, use no_constraint() type if no constraint is required"))
        elseif !(n_dists == n_names)
            throw(DimensionMismatch("There must be one name per parameter distribution"))
        else
            constraints = cat(constraints..., dims = 1)

            new{PDType, ConstraintType, ST}(parameter_distributions, constraints, names)
        end
    end

end



## Functions

"""
    get_name(pd::ParameterDistribution)

Returns a list of ParameterDistribution names.
"""
get_name(pd::ParameterDistribution) = pd.names

"""
    get_dimensions(pd::ParameterDistribution)

The number of dimensions of the parameter space.
"""
function get_dimensions(pd::ParameterDistribution)
    return [dimension(d) for d in pd.distributions]
end

function get_total_dimension(pd::ParameterDistribution)
    return sum(dimension(d) for d in pd.distributions)
end

"""
    get_n_samples(pd::ParameterDistribution)

The number of samples in a Samples distribution
"""
function get_n_samples(pd::ParameterDistribution)
    return Dict{String, Any}(pd.names[i] => n_samples(d) for (i, d) in enumerate(pd.distributions))
end

"""
    get_all_constraints(pd::ParameterDistribution)

Returns the (flattened) array of constraints of the parameter distribution.
"""
get_all_constraints(pd::ParameterDistribution) = pd.constraints

"""
    batch(pd::ParameterDistribution)

Returns a list of contiguous `[collect(1:i), collect(i+1:j),... ]` used to split parameter arrays by distribution dimensions.
"""
function batch(pd::ParameterDistribution)
    #chunk xarray to give to the different distributions.
    d_dim = get_dimensions(pd) #e.g [4,1,2]
    d_dim_tmp = Array{Int64}(undef, size(d_dim)[1] + 1)
    d_dim_tmp[1] = 0
    for i in 2:(size(d_dim)[1] + 1)
        d_dim_tmp[i] = sum(d_dim[1:(i - 1)]) # e.g [0,4,5,7]
    end

    return [collect((d_dim_tmp[i] + 1):d_dim_tmp[i + 1]) for i in 1:size(d_dim)[1]] # e.g [1:4, 5:5, 6:7]
end

"""
    get_distribution(pd::ParameterDistribution)

Returns a `Dict` of `ParameterDistribution` distributions, with the parameter names
as dictionary keys. For parameters represented by `Samples`, the samples are returned
as a 2D (`parameter_dimension x n_samples`) array.
"""
function get_distribution(pd::ParameterDistribution)
    return Dict{String, Any}(pd.names[i] => get_distribution(d) for (i, d) in enumerate(pd.distributions))
end

get_distribution(d::Samples) = d.distribution_samples

get_distribution(d::Parameterized) = d.distribution

"""
    sample_distribution([rng], pd::ParameterDistribution, [n_draws])

Draws `n_draws` samples from the parameter distributions `pd`. Returns an array, with 
parameters as columns. `rng` is optional and defaults to `Random.GLOBAL_RNG`. `n_draws` is 
optional and defaults to 1. 
"""
function sample_distribution(rng::AbstractRNG, pd::ParameterDistribution, n_draws::IT) where {IT <: Integer}
    return cat([sample_distribution(rng, d, n_draws) for d in pd.distributions]..., dims = 1)
end

# define methods that dispatch to the above with Random.GLOBAL_RNG as a default value for rng
sample_distribution(pd::ParameterDistribution, n_draws::IT) where {IT <: Integer} =
    sample_distribution(Random.GLOBAL_RNG, pd, n_draws)
sample_distribution(rng::AbstractRNG, pd::ParameterDistribution) = sample_distribution(rng, pd, 1)
sample_distribution(pd::ParameterDistribution) = sample_distribution(Random.GLOBAL_RNG, pd, 1)

"""
    sample_distribution([rng], d::Samples, [n_draws])

Draws `n_draws` samples from the parameter distributions `d`. Returns an array, with 
parameters as columns. `rng` is optional and defaults to `Random.GLOBAL_RNG`. `n_draws` is 
optional and defaults to 1. 
"""
function sample_distribution(rng::AbstractRNG, d::Samples, n_draws::IT) where {IT <: Integer}
    n_stored_samples = n_samples(d)
    samples_idx = sample(rng, collect(1:n_stored_samples), n_draws)
    if dimension(d) == 1
        return reshape(d.distribution_samples[:, samples_idx], :, n_draws) #columns are parameters
    else
        return d.distribution_samples[:, samples_idx]
    end
end

# define methods that dispatch to the above with Random.GLOBAL_RNG as a default value for rng
sample_distribution(d::Samples, n_draws::IT) where {IT <: Integer} = sample_distribution(Random.GLOBAL_RNG, d, n_draws)
sample_distribution(rng::AbstractRNG, d::Samples) = sample_distribution(rng, d, 1)
sample_distribution(d::Samples) = sample_distribution(Random.GLOBAL_RNG, d, 1)

"""
    sample_distribution([rng], d::Parameterized, [n_draws])

Draws `n_draws` samples from the parameter distributions `d`. Returns an array, with 
parameters as columns. `rng` is optional and defaults to `Random.GLOBAL_RNG`. `n_draws` is 
optional and defaults to 1. 
"""
function sample_distribution(rng::AbstractRNG, d::Parameterized, n_draws::IT) where {IT <: Integer}
    if dimension(d) == 1
        return reshape(rand(rng, d.distribution, n_draws), :, n_draws) #columns are parameters
    else
        return rand(rng, d.distribution, n_draws)
    end
end

# define methods that dispatch to the above with Random.GLOBAL_RNG as a default value for rng
sample_distribution(d::Parameterized, n_draws::IT) where {IT <: Integer} =
    sample_distribution(Random.GLOBAL_RNG, d, n_draws)
sample_distribution(rng::AbstractRNG, d::Parameterized) = sample_distribution(rng, d, 1)
sample_distribution(d::Parameterized) = sample_distribution(Random.GLOBAL_RNG, d, 1)

"""
    logpdf(pd::ParameterDistribution, xarray::Array{<:Real,1})

Obtains the independent logpdfs of the parameter distributions at `xarray`
(non-Samples Distributions only), and returns their sum.
"""
get_logpdf(d::Parameterized, xarray::Array{FT, 1}) where {FT <: Real} = logpdf.(d.distribution, xarray)

function get_logpdf(pd::ParameterDistribution, xarray::Array{FT, 1}) where {FT <: Real}
    #first check we don't have sampled distribution
    for d in pd.distributions
        if typeof(d) <: Samples
            throw(ErrorException("Cannot compute get_logpdf of Samples distribution. Consider using a Parameterized type for your prior."))
        end
    end
    #assert xarray correct dim/length
    if size(xarray)[1] != get_total_dimension(pd)
        throw(DimensionMismatch("xarray must have dimension equal to the parameter space"))
    end

    # get the index of xarray chunks to give to the different distributions.
    batches = batch(pd)

    # perform the logpdf of each of the distributions, and returns their sum    
    return sum(cat([get_logpdf(d, xarray[batches[i]]) for (i, d) in enumerate(pd.distributions)]..., dims = 1))
end

#extending StatsBase cov,var
"""
    var(pd::ParameterDistribution)
Returns a flattened variance of the distributions
"""
var(d::Parameterized) = var(d.distribution)
var(d::Samples) = var(d.distribution_samples)
function var(pd::ParameterDistribution)
    d_dims = get_dimensions(pd)
    block_var = Array{Any}(size(d_dims)[1])
    
    for (i, dimension) in enumerate(d_dims)
            block_var[i] = var(pd.distributions[i])
    end
    return cat(block_var..., dims = 1) #build the flattened vector
    
end

"""
    cov(pd::ParameterDistribution)

Returns a dense blocked (co)variance of the distributions.
"""
cov(d::Parameterized) = cov(d.distribution)
cov(d::Samples) = cov(d.distribution_samples, dims = 2) #parameters are columns
function cov(pd::ParameterDistribution)
    d_dims = get_dimensions(pd)
    
    # create each block (co)variance
    block_cov = Array{Any}(undef, size(d_dims)[1])
    for (i, dimension) in enumerate(d_dims)
        if dimension == 1
            block_cov[i] = var(pd.distributions[i])
        else
            block_cov[i] = cov(pd.distributions[i])
        end
    end

    return cat(block_cov..., dims = (1, 2)) #build the block diagonal (dense) matrix

end

#extending mean
"""
    mean(pd::ParameterDistribution)

Returns a concatenated mean of the parameter distributions. 
"""
mean(d::Parameterized) = mean(d.distribution)
mean(d::Samples) = mean(d.distribution_samples, dims = 2)
function mean(pd::ParameterDistribution)
    return cat([mean(d) for d in pd.distributions]..., dims = 1)
end

#apply transforms

"""
    transform_constrained_to_unconstrained(pd::ParameterDistribution, xarray::Array{<:Real,1})

Apply the transformation to map (possibly constrained) parameters `xarray` into the unconstrained space.
"""
function transform_constrained_to_unconstrained(pd::ParameterDistribution, xarray::Array{FT, 1}) where {FT <: Real}
    return cat([c.constrained_to_unconstrained(xarray[i]) for (i, c) in enumerate(pd.constraints)]..., dims = 1)
end

"""
    transform_constrained_to_unconstrained(pd::ParameterDistribution, xarray::Array{<:Real,2})

Apply the transformation to map (possibly constrained) parameter samples `xarray` into the unconstrained space.
Here, `xarray` contains parameters as columns and samples as rows.
"""
function transform_constrained_to_unconstrained(pd::ParameterDistribution, xarray::Array{FT, 2}) where {FT <: Real}
    return Array(hcat([c.constrained_to_unconstrained.(xarray[i, :]) for (i, c) in enumerate(pd.constraints)]...)')
end

"""
    transform_unconstrained_to_constrained(pd::ParameterDistribution, xarray::Array{<:Real,1})

Apply the transformation to map parameters `xarray` from the unconstrained space into (possibly constrained) space.
"""
function transform_unconstrained_to_constrained(pd::ParameterDistribution, xarray::Array{FT, 1}) where {FT <: Real}
    return cat([c.unconstrained_to_constrained(xarray[i]) for (i, c) in enumerate(pd.constraints)]..., dims = 1)
end

"""
    transform_unconstrained_to_constrained(pd::ParameterDistribution, xarray::Array{<:Real,2})

Apply the transformation to map parameter samples `xarray` from the unconstrained space into (possibly constrained) space.
Here, `xarray` contains parameters as columns and samples as rows.
"""
function transform_unconstrained_to_constrained(pd::ParameterDistribution, xarray::Array{FT, 2}) where {FT <: Real}
    return Array(hcat([c.unconstrained_to_constrained.(xarray[i, :]) for (i, c) in enumerate(pd.constraints)]...)')
end

"""
    transform_unconstrained_to_constrained(pd::ParameterDistribution, xarray::Array{Array{<:Real,2},1})

Apply the transformation to map parameter sample ensembles `xarray` from the unconstrained space into (possibly constrained) space.
Here, `xarray` contains parameters sample ensembles for different EKP iterations.
"""
function transform_unconstrained_to_constrained(
    pd::ParameterDistribution,
    xarray::Array{Array{FT, 2}, 1},
) where {FT <: Real}
    transf_xarray = []
    for elem in xarray
        push!(transf_xarray, transform_unconstrained_to_constrained(pd, elem))
    end
    return transf_xarray
end

end # of module
