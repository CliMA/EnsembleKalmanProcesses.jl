module ParameterDistributions

## Usings
using Distributions
using Statistics
using Random
using Optim, QuadGK

#import (to add definitions)
import StatsBase: mean, var, cov, sample
import Base: size, length, ndims
## Exports

#types
export ParameterDistributionType
export ConstraintType

#objects
export Parameterized, Samples, VectorOfParameterized
export ParameterDistribution
export Constraint

#functions
export get_name, get_distribution, ndims, get_dimensions, get_all_constraints, get_n_samples
export sample
export no_constraint, bounded_below, bounded_above, bounded
export transform_constrained_to_unconstrained, transform_unconstrained_to_constrained
export get_logpdf, batch

export combine_distributions
export constrained_gaussian
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
    distribution_samples::AbstractMatrix{FT} #parameters are columns
    Samples(distribution_samples::AbstractMatrix{FT}; params_are_columns = true) where {FT <: Real} =
        params_are_columns ? new{FT}(distribution_samples) : new{FT}(permutedims(distribution_samples, (2, 1)))
    #Distinguish 1 sample of an ND parameter or N samples of 1D parameter, and store as 2D array  
    Samples(distribution_samples::AbstractVector{FT}; params_are_columns = true) where {FT <: Real} =
        params_are_columns ? new{FT}(reshape(distribution_samples, 1, :)) : new{FT}(reshape(distribution_samples, :, 1))
end

"""
    VectorOfParameterized <: ParameterDistributionType

A distribution built from an array of Parametrized distributions.
A utility to help stacking of distributions where a multivariate equivalent doesn't exist
"""
struct VectorOfParameterized{DT <: Distribution} <: ParameterDistributionType
    distribution::AbstractVector{DT}
end


# For the transforms
abstract type ConstraintType end

"""
    Constraint <: ConstraintType

Class describing a 1D bijection between constrained and unconstrained spaces.
"""
struct Constraint <: ConstraintType
    constrained_to_unconstrained::Function
    c_to_u_jacobian::Function
    unconstrained_to_constrained::Function
end


"""
    no_constraint()

Constructs a Constraint with no constraints, enforced by maps x -> x and x -> x.
"""
function no_constraint()
    c_to_u = (x -> x)
    jacobian = (x -> 1.0)
    u_to_c = (x -> x)
    return Constraint(c_to_u, jacobian, u_to_c)
end

"""
    bounded_below(lower_bound::FT) where {FT <: Real}

Constructs a Constraint with provided lower bound, enforced by maps `x -> log(x - lower_bound)`
and `x -> exp(x) + lower_bound`.
"""
function bounded_below(lower_bound::FT) where {FT <: Real}
    if isinf(lower_bound)
        return no_constraint()
    end
    c_to_u = (x -> log(x - lower_bound))
    jacobian = (x -> 1.0 / (x - lower_bound))
    u_to_c = (x -> exp(x) + lower_bound)
    return Constraint(c_to_u, jacobian, u_to_c)
end

"""
    bounded_above(upper_bound::FT) where {FT <: Real} 

Constructs a Constraint with provided upper bound, enforced by maps `x -> log(upper_bound - x)`
and `x -> upper_bound - exp(x)`.
"""
function bounded_above(upper_bound::FT) where {FT <: Real}
    if isinf(upper_bound)
        return no_constraint()
    end
    c_to_u = (x -> -log(upper_bound - x))
    jacobian = (x -> 1.0 / (upper_bound - x))
    u_to_c = (x -> upper_bound - exp(-x))
    return Constraint(c_to_u, jacobian, u_to_c)
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
    # As far as I know, only way to dispatch method based on isinf() would be to bring in 
    # Traits as another dependency, which would be overkill
    if isinf(lower_bound)
        if isinf(upper_bound)
            return no_constraint()
        else
            return bounded_above(upper_bound)
        end
    else
        if isinf(upper_bound)
            return bounded_below(lower_bound)
        else
            c_to_u = (x -> log((x - lower_bound) / (upper_bound - x)))
            jacobian = (x -> 1.0 / (upper_bound - x) + 1.0 / (x - lower_bound))
            u_to_c = (x -> upper_bound - (upper_bound - lower_bound) / (exp(x) + 1))
            return Constraint(c_to_u, jacobian, u_to_c)
        end
    end
end

#extending Base.length
"""
    length(c<:ConstraintType)

A constraint has length 1. 
"""
length(c::CType) where {CType <: ConstraintType} = length([c])

#extending Base.size
"""
    size(c<:ConstraintType)

A constraint has size 1.
"""
size(c::CType) where {CType <: ConstraintType} = size([c])

"""
    ndims(d<:ParametrizedDistributionType)

The number of dimensions of the parameter space
"""
ndims(d::Parameterized) = length(d.distribution)

ndims(d::Samples) = size(d.distribution_samples, 1)

ndims(d::VectorOfParameterized) = sum(length.(d.distribution))
"""
    n_samples(d<:Samples)

The number of samples in the array.
"""
n_samples(d::Samples) = size(d.distribution_samples)[2]

n_samples(d::Parameterized) = "Distribution stored in Parameterized form, draw samples using `sample` function"

n_samples(d::VectorOfParameterized) = "Distribution stored in Parameterized form, draw samples using `sample` function"


"""
    ParameterDistribution

Structure to hold a parameter distribution, always stored as an array internally.
"""
struct ParameterDistribution{PDType <: ParameterDistributionType, CType <: ConstraintType, ST <: AbstractString}
    distribution::AbstractVector{PDType}
    constraint::AbstractVector{CType}
    name::AbstractVector{ST}
end

"""
    ParameterDistribution(param_dist_dict::Union{Dict,AbstractVector})

Constructor taking in a Dict or array of Dicts. Each dict must contain the key-val pairs:
- `"distribution"` - a distribution of `ParameterDistributionType`
- `"constraint"` - constraint(s) given as a `ConstraintType` or array of `ConstraintType`s with length equal to the dims of the distribution
- `"name"` - a name of the distribution as a String.
"""
function ParameterDistribution(param_dist_dict::Union{Dict, AbstractVector})

    #check type
    if !(isa(param_dist_dict, Dict) || eltype(param_dist_dict) <: Dict)
        throw(ArgumentError("input argument must be a Dict, or <:AbstractVector{Dict}"))
    end

    # make copy as array
    param_dist_dict_array = !isa(param_dist_dict, AbstractVector) ? [param_dist_dict] : param_dist_dict
    # perform checks on the individual distributions
    for pdd in param_dist_dict_array
        # check all keys are present
        if !all(["distribution", "name", "constraint"] .∈ [collect(keys(pdd))])
            throw(
                ArgumentError("input dictionaries must contain the keys: \"distribution\", \"name\", \"constraint\" "),
            )
        end

        distribution = pdd["distribution"]
        name = pdd["name"]
        constraint = pdd["constraint"]

        # check key types
        if !isa(distribution, ParameterDistributionType)
            throw(
                ArgumentError(
                    "Value of \"distribution\" must be a valid ParameterDistribution object: Parameterized or Samples",
                ),
            )
        end
        if !isa(constraint, ConstraintType)
            if !isa(constraint, AbstractVector) #it's not a vector either
                throw(
                    ArgumentError(
                        "Value of \"constraint\" must be a ConstraintType, or <:AbstractVector(ConstraintType)",
                    ),
                )
            elseif !(eltype(constraint) <: ConstraintType) #it is a vector, but not of constraint
                throw(
                    ArgumentError(
                        "Value of \"constraint\" must be a ConstraintType, or <:AbstractVector(ConstraintType)",
                    ),
                )
            end
        end
        if !isa(name, String)
            throw(ArgumentError("Value of \"name\" must be a String"))
        end

        # 1 constraint per dimension check
        constraint_array = isa(constraint, ConstraintType) ? [constraint] : constraint
        n_parameters = ndims(distribution)

        if !(n_parameters == length(constraint_array))
            throw(
                DimensionMismatch(
                    "There must be one constraint dimension in a parameter distribution, use no_constraint() type if no constraint is required",
                ),
            )
        end
    end

    # flatten the structure
    distribution = getindex.(param_dist_dict_array, "distribution")
    flat_constraint = cat(getindex.(param_dist_dict_array, "constraint")..., dims = 1)
    name = getindex.(param_dist_dict_array, "name")

    # build the object
    return ParameterDistribution(distribution, flat_constraint, name)

end

"""
    function ParameterDistribution(distribution::ParameterDistributionType,
                                   constraint::Union{ConstraintType,AbstractVector{ConstraintType}},
                                   name::AbstractString)

constructor of a ParameterDistribution from a single `distribution`, (array of) `constraint`, `name`.
these can used to build another ParameterDistribution
"""
function ParameterDistribution(
    distribution::ParameterDistributionType,
    constraint::Union{ConstraintType, AbstractVector},
    name::AbstractString,
)

    if !(typeof(constraint) <: ConstraintType || eltype(constraint) <: ConstraintType) # if it is a vector, but not of constraint
        throw(ArgumentError("`constraint` must be a ConstraintType, or Vector of ConstraintType's "))
    end
    # 1 constraint per dimension check
    constraint_vec = isa(constraint, ConstraintType) ? [constraint] : constraint
    n_parameters = ndims(distribution)

    if !(n_parameters == length(constraint_vec))
        throw(
            DimensionMismatch(
                "There must be one constraint dimension in a parameter distribution, use no_constraint() type if no constraint is required",
            ),
        )
    end

    # flatten the structure
    distribution_vec = [distribution]
    flat_constraint_vec = cat(constraint_vec..., dims = 1)
    name_vec = [name]

    # build the object
    return ParameterDistribution(distribution_vec, flat_constraint_vec, name_vec)

end

"""
    combine_distributions(pd_vec::AbstractVector{ParameterDistribution}...; dims = 1)

concatenate from vector of single ParameterDistributions, dims keyword is unused
"""
function combine_distributions(pd_vec::AbstractVector{PD}; dims = 1) where {PD <: ParameterDistribution}
    # flatten the structure
    distribution = cat([d.distribution for d in pd_vec]..., dims = 1)
    constraint = cat([d.constraint for d in pd_vec]..., dims = 1)
    name = cat([d.name for d in pd_vec]..., dims = 1)
    return ParameterDistribution(distribution, constraint, name)
end

## Functions

"""
    get_name(pd::ParameterDistribution)

Returns a list of ParameterDistribution names.
"""
get_name(pd::ParameterDistribution) = pd.name

"""
    get_dimensions(pd::ParameterDistribution)

The number of dimensions of the parameter space.
"""
function get_dimensions(pd::ParameterDistribution)
    return [ndims(d) for d in pd.distribution]
end
function get_dimensions(d::VectorOfParameterized)
    return [length(dd) for dd in d.distribution]
end

function ndims(pd::ParameterDistribution)
    return sum(get_dimensions(pd))
end

"""
    get_n_samples(pd::ParameterDistribution)

The number of samples in a Samples distribution
"""
function get_n_samples(pd::ParameterDistribution)
    return Dict{String, Any}(pd.name[i] => n_samples(d) for (i, d) in enumerate(pd.distribution))
end

"""
    get_all_constraints(pd::ParameterDistribution)

Returns the (flattened) array of constraints of the parameter distribution.
"""
get_all_constraints(pd::ParameterDistribution) = pd.constraint

"""
    batch(pd::ParameterDistribution)

Returns a list of contiguous `[collect(1:i), collect(i+1:j),... ]` used to split parameter arrays by distribution dimensions.
"""
function batch(pd::Union{ParameterDistribution, VectorOfParameterized})
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
    return Dict{String, Any}(pd.name[i] => get_distribution(d) for (i, d) in enumerate(pd.distribution))
end

get_distribution(d::Samples) = d.distribution_samples

get_distribution(d::Parameterized) = d.distribution

get_distribution(d::VectorOfParameterized) = d.distribution

"""
    sample([rng], pd::ParameterDistribution, [n_draws])

Draws `n_draws` samples from the parameter distributions `pd`. Returns an array, with 
parameters as columns. `rng` is optional and defaults to `Random.GLOBAL_RNG`. `n_draws` is 
optional and defaults to 1. 
"""
function sample(rng::AbstractRNG, pd::ParameterDistribution, n_draws::IT) where {IT <: Integer}
    return cat([sample(rng, d, n_draws) for d in pd.distribution]..., dims = 1)
end

# define methods that dispatch to the above with Random.GLOBAL_RNG as a default value for rng
sample(pd::ParameterDistribution, n_draws::IT) where {IT <: Integer} = sample(Random.GLOBAL_RNG, pd, n_draws)
sample(rng::AbstractRNG, pd::ParameterDistribution) = sample(rng, pd, 1)
sample(pd::ParameterDistribution) = sample(Random.GLOBAL_RNG, pd, 1)

"""
    sample([rng], d::Samples, [n_draws])

Draws `n_draws` samples from the parameter distributions `d`. Returns an array, with 
parameters as columns. `rng` is optional and defaults to `Random.GLOBAL_RNG`. `n_draws` is 
optional and defaults to 1. 
"""
function sample(rng::AbstractRNG, d::Samples, n_draws::IT) where {IT <: Integer}
    n_stored_samples = n_samples(d)
    samples_idx = sample(rng, collect(1:n_stored_samples), n_draws)
    if ndims(d) == 1
        return reshape(d.distribution_samples[:, samples_idx], :, n_draws) #columns are parameters
    else
        return d.distribution_samples[:, samples_idx]
    end
end

# define methods that dispatch to the above with Random.GLOBAL_RNG as a default value for rng
sample(d::Samples, n_draws::IT) where {IT <: Integer} = sample(Random.GLOBAL_RNG, d, n_draws)
sample(rng::AbstractRNG, d::Samples) = sample(rng, d, 1)
sample(d::Samples) = sample(Random.GLOBAL_RNG, d, 1)

"""
    sample([rng], d::Parameterized, [n_draws])

Draws `n_draws` samples from the parameter distributions `d`. Returns an array, with 
parameters as columns. `rng` is optional and defaults to `Random.GLOBAL_RNG`. `n_draws` is 
optional and defaults to 1. 
"""
function sample(rng::AbstractRNG, d::Parameterized, n_draws::IT) where {IT <: Integer}
    if ndims(d) == 1
        return reshape(rand(rng, d.distribution, n_draws), :, n_draws) #columns are parameters
    else
        return rand(rng, d.distribution, n_draws)
    end
end

# define methods that dispatch to the above with Random.GLOBAL_RNG as a default value for rng
sample(d::Parameterized, n_draws::IT) where {IT <: Integer} = sample(Random.GLOBAL_RNG, d, n_draws)
sample(rng::AbstractRNG, d::Parameterized) = sample(rng, d, 1)
sample(d::Parameterized) = sample(Random.GLOBAL_RNG, d, 1)

"""
    sample([rng], d::VectorOfParameterized, [n_draws])

Draws `n_draws` samples from the parameter distributions `d`. Returns an array, with 
parameters as columns. `rng` is optional and defaults to `Random.GLOBAL_RNG`. `n_draws` is 
optional and defaults to 1. 
"""
function sample(rng::AbstractRNG, d::VectorOfParameterized, n_draws::IT) where {IT <: Integer}
    samples = zeros(ndims(d), n_draws)
    batches = batch(d)
    dimensions = get_dimensions(d)
    for (i, dd) in enumerate(d.distribution)
        samples[batches[i], :] = rand(rng, dd, n_draws) #columns are parameters
    end
    return samples
end

sample(d::VectorOfParameterized, n_draws::IT) where {IT <: Integer} = sample(Random.GLOBAL_RNG, d, n_draws)
sample(rng::AbstractRNG, d::VectorOfParameterized) = sample(rng, d, 1)
sample(d::VectorOfParameterized) = sample(Random.GLOBAL_RNG, d, 1)

"""
    logpdf(pd::ParameterDistribution, xarray::Array{<:Real,1})

Obtains the independent logpdfs of the parameter distributions at `xarray`
(non-Samples Distributions only), and returns their sum.
"""
get_logpdf(d::Parameterized, xarray::AbstractVector{FT}) where {FT <: Real} = logpdf.(d.distribution, xarray)

function get_logpdf(d::VectorOfParameterized, xarray::AbstractVector{FT}) where {FT <: Real}
    # get the index of xarray chunks to give to the different distributions.
    batches = batch(d)
    dimensions = get_dimensions(d)
    lpdfsum = 0.0
    # perform the logpdf of each of the distributions, and returns their sum    
    for (i, dd) in enumerate(d.distribution)
        if dimensions[i] == 1
            lpdfsum += logpdf.(dd, xarray[batches[i]])[1]
        else
            lpdfsum += logpdf(dd, xarray[batches[i]])
        end
    end
    return lpdfsum
end

function get_logpdf(pd::ParameterDistribution, xarray::AbstractVector{FT}) where {FT <: Real}
    #first check we don't have sampled distribution
    for d in pd.distribution
        if typeof(d) <: Samples
            throw(
                ErrorException(
                    "Cannot compute get_logpdf of Samples distribution. Consider using a Parameterized type for your prior.",
                ),
            )
        end
    end
    #assert xarray correct dim/length
    if size(xarray)[1] != ndims(pd)
        throw(DimensionMismatch("xarray must have dimension equal to the parameter space"))
    end

    # get the index of xarray chunks to give to the different distributions.
    batches = batch(pd)

    # perform the logpdf of each of the distributions, and returns their sum    
    return sum(cat([get_logpdf(d, xarray[batches[i]]) for (i, d) in enumerate(pd.distribution)]..., dims = 1))
end

#extending StatsBase cov,var
"""
    var(pd::ParameterDistribution)
Returns a flattened variance of the distributions
"""
var(d::Parameterized) = var(d.distribution)
var(d::Samples) = var(d.distribution_samples, dims = 2)
function var(d::VectorOfParameterized)
    d_dims = get_dimensions(d)
    block_var = Array{Any}(undef, size(d_dims)[1])

    for (i, dimension) in enumerate(d_dims)
        block_var[i] = var(d.distribution[i])
    end
    return cat(block_var..., dims = 1)
end

function var(pd::ParameterDistribution)
    d_dims = get_dimensions(pd)
    block_var = Array{Any}(undef, size(d_dims)[1])

    for (i, dimension) in enumerate(d_dims)
        block_var[i] = var(pd.distribution[i])
    end
    return cat(block_var..., dims = 1) #build the flattened vector
end


"""
    cov(pd::ParameterDistribution)

Returns a dense blocked (co)variance of the distributions.
"""
cov(d::Parameterized) = cov(d.distribution)
cov(d::Samples) = cov(d.distribution_samples, dims = 2) #parameters are columns
function cov(d::VectorOfParameterized)
    d_dims = get_dimensions(d)

    # create each block (co)variance
    block_cov = Array{Any}(undef, size(d_dims)[1])
    for (i, dimension) in enumerate(d_dims)
        if dimension == 1
            block_cov[i] = var(d.distribution[i])
        else
            block_cov[i] = cov(d.distribution[i])
        end
    end

    return cat(block_cov..., dims = (1, 2)) #build the block diagonal (dense) matrix

end

function cov(pd::ParameterDistribution)
    d_dims = get_dimensions(pd)

    # create each block (co)variance
    block_cov = Array{Any}(undef, size(d_dims)[1])
    for (i, dimension) in enumerate(d_dims)
        if dimension == 1
            block_cov[i] = var(pd.distribution[i])
        else
            block_cov[i] = cov(pd.distribution[i])
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
function mean(d::VectorOfParameterized)
    return cat([mean(dd) for dd in d.distribution]..., dims = 1)
end
function mean(pd::ParameterDistribution)
    return cat([mean(d) for d in pd.distribution]..., dims = 1)
end

#apply transforms

"""
    transform_constrained_to_unconstrained(pd::ParameterDistribution, xarray::Array{<:Real,1})

Apply the transformation to map (possibly constrained) parameters `xarray` into the unconstrained space.
"""
function transform_constrained_to_unconstrained(
    pd::ParameterDistribution,
    xarray::AbstractVector{FT},
) where {FT <: Real}
    return cat([c.constrained_to_unconstrained(xarray[i]) for (i, c) in enumerate(pd.constraint)]..., dims = 1)
end

"""
    transform_constrained_to_unconstrained(pd::ParameterDistribution, xarray::Array{<:Real,2})

Apply the transformation to map (possibly constrained) parameter samples `xarray` into the unconstrained space.
Here, `xarray` contains parameters as columns and samples as rows.
"""
function transform_constrained_to_unconstrained(
    pd::ParameterDistribution,
    xarray::AbstractMatrix{FT},
) where {FT <: Real}
    return Array(hcat([c.constrained_to_unconstrained.(xarray[i, :]) for (i, c) in enumerate(pd.constraint)]...)')
end

"""
    transform_unconstrained_to_constrained(pd::ParameterDistribution, xarray::Array{<:Real,1})

Apply the transformation to map parameters `xarray` from the unconstrained space into (possibly constrained) space.
"""
function transform_unconstrained_to_constrained(
    pd::ParameterDistribution,
    xarray::AbstractVector{FT},
) where {FT <: Real}
    return cat([c.unconstrained_to_constrained(xarray[i]) for (i, c) in enumerate(pd.constraint)]..., dims = 1)
end

"""
    transform_unconstrained_to_constrained(pd::ParameterDistribution, xarray::Array{<:Real,2})

Apply the transformation to map parameter samples `xarray` from the unconstrained space into (possibly constrained) space.
Here, `xarray` contains parameters as columns and samples as rows.
"""
function transform_unconstrained_to_constrained(
    pd::ParameterDistribution,
    xarray::AbstractMatrix{FT},
) where {FT <: Real}
    return Array(hcat([c.unconstrained_to_constrained.(xarray[i, :]) for (i, c) in enumerate(pd.constraint)]...)')
end

"""
    transform_unconstrained_to_constrained(pd::ParameterDistribution, xarray::Array{Array{<:Real,2},1})

Apply the transformation to map parameter sample ensembles `xarray` from the unconstrained space into (possibly constrained) space.
Here, `xarray` is an iterable of parameters sample ensembles for different EKP iterations.
"""
function transform_unconstrained_to_constrained(
    pd::ParameterDistribution,
    xarray, # ::Iterable{AbstractMatrix{FT}},
) where {FT <: Real}
    transf_xarray = []
    for elem in xarray
        push!(transf_xarray, transform_unconstrained_to_constrained(pd, elem))
    end
    return transf_xarray
end

# -------------------------------------------------------------------------------------
# constructor for numerically optimized constrained distributions

function _moment(m::Integer, d::UnivariateDistribution, c::Constraint)
    # Rough (hopefully fast) 1D numerical integration of constrained distribution expectation
    # values. Integrate in constrained space, although constraints can give singular behavior 
    # near bounds.
    # Intended use is only on bounded integrals; otherwise run into 
    # https://github.com/JuliaMath/QuadGK.jl/issues/38
    min = c.unconstrained_to_constrained(minimum(d))
    max = c.unconstrained_to_constrained(maximum(d))
    function integrand(x)
        log_pdf = logpdf(d, c.constrained_to_unconstrained(x))
        # jacobian always >= 0; use logs to avoid over/underflow
        return isinf(log_pdf) ? 0.0 : x^m * exp(log(c.c_to_u_jacobian(x)) + log_pdf)
    end
    return quadgk(integrand, min, max, order = 9, rtol = 1e-5, atol = 1e-6)[1]
end
function _mean_std(μ::FT, σ::FT, c::Constraint) where {FT <: Real}
    d = Normal(μ, σ)
    m = [_moment(k, d, c) for k in 1:2]
    return (m[1], sqrt(m[2] - m[1]^2))
end
function _lognormal_mean_std(μ_u::FT, σ_u::FT) where {FT <: Real}
    # known analytic solution for lognormal distribution
    return (exp(μ_u + σ_u^2 / 2.0), exp(μ_u + σ_u^2 / 2.0) * sqrt(expm1(σ_u^2)))
end
function _inverse_lognormal_mean_std(μ_c::FT, σ_c::FT) where {FT <: Real}
    # known analytic solution for lognormal distribution
    return (log(μ_c) - 0.5 * log1p((σ_c / μ_c)^2), sqrt(log1p((σ_c / μ_c)^2)))
end

"""
    constrained_gaussian(name::AbstractString, μ_c::FT, σ_c::FT, lower_bound::FT, upper_bound::FT)

Constructor for a 1D ParameterDistribution consisting of a transformed Gaussian, constrained
to have support on [`lower_bound`, `upper_bound`], with first two moments μ_c and σ_c^2. The 
moment integrals can't be done in closed form, so we set the parameters of the distribution
with numerical optimization.

!!! note
    The intended use case is defining priors set from user expertise for use in inference 
    with adequate data, so for the sake of performance we only require that the optimization
    reproduce μ_c, σ_c to a loose tolerance (1e-5). Warnings are logged when the optimization
    fails.

!!! note
    The distribution may be bimodal for σ_c large relative to the width of the bound interval.
    In extreme cases the distribution becomes concentrated at the bound endpoints. We regard
    this as a feature, not a bug, and do not warn the user when bimodality occurs.
"""
function constrained_gaussian(
    name::AbstractString,
    μ_c::FT,
    σ_c::FT,
    lower_bound::FT,
    upper_bound::FT;
    optim_algorithm::Optim.AbstractOptimizer = NelderMead(),
    optim_kwargs...,
) where {FT <: Real}
    if (upper_bound <= lower_bound)
        throw(
            DomainError(
                "`$(name)`: Upper bound must be greater than lower bound (got [$(lower_bound), $(upper_bound)])",
            ),
        )
    end
    if (μ_c <= lower_bound) || (μ_c >= upper_bound)
        throw(DomainError("`$(name)`: Target mean $(μ_c) must be within constraint [$(lower_bound), $(upper_bound)]"))
    end

    if isinf(lower_bound)
        if isinf(upper_bound)
            μ_u, σ_u = (μ_c, σ_c)
        else
            # linear change of variables in integral, std unchanged
            μ_u, σ_u = _inverse_lognormal_mean_std(upper_bound - μ_c, σ_c)
        end
    else
        if isinf(upper_bound)
            # linear change of variables in integral, std unchanged
            μ_u, σ_u = _inverse_lognormal_mean_std(μ_c - lower_bound, σ_c)
        else
            # finite interval case; need to solve numerically
            if (μ_c - 0.7 * σ_c <= lower_bound)
                throw(DomainError("`$(name)`: Target std $(σ_c) puts μ - σ too close to lower bound $(lower_bound)"))
            end
            if (μ_c + 0.7 * σ_c >= upper_bound)
                throw(DomainError("`$(name)`: Target std $(σ_c) puts μ + σ too close to upper bound $(upper_bound)"))
            end
            μ_u, σ_u = _constrained_gaussian(
                name,
                μ_c,
                σ_c,
                lower_bound,
                upper_bound;
                optim_algorithm = optim_algorithm,
                optim_kwargs...,
            )
        end
    end
    cons = bounded(lower_bound, upper_bound)
    return ParameterDistribution(Parameterized(Normal(μ_u, σ_u)), cons, name)
end

function _constrained_gaussian_guess(μ_c::FT, σ_c::FT, lower_bound::FT, upper_bound::FT) where {FT <: Real}
    # Initial guess for μ_c, σ_c in _constrained_gaussian optimization.
    # Equations result from inverting the system 
    # μ_c = f^{-1}(μ_u / sqrt(1 + σ_u^2/2))
    # σ_c = f^{-1}((μ_u + σ_u/2) / sqrt(1 + σ_u^2/2)) - f^{-1}((μ_u - σ_u/2) / sqrt(1 + σ_u^2/2))
    # where f^{-1} is the specific form of cons.unconstrained_to_constrained().

    cons = bounded(lower_bound, upper_bound)
    _Δul = upper_bound - lower_bound
    _Δum = upper_bound - μ_c
    _Δml = μ_c - lower_bound
    init_σ_u =
        0.5 * (
            -σ_c * (_Δum^2 + _Δml^2) / (_Δum * _Δml * (σ_c - 1.0)) +
            hypot(2 * _Δum * _Δml, _Δul * σ_c * (_Δum - _Δml)) / (_Δum * _Δml * abs(σ_c - 1.0))
        )
    init_σ_u = 2.0 * log(init_σ_u) / sqrt(1.0 - 2 * log(init_σ_u)^2)
    init_μ_u = sqrt(1.0 + init_σ_u^2 / 2.0) * cons.constrained_to_unconstrained(μ_c)
    return (init_μ_u, init_σ_u)
end

function _constrained_gaussian(
    name::AbstractString,
    μ_c::FT,
    σ_c::FT,
    lower_bound::FT,
    upper_bound::FT;
    optim_algorithm::Optim.AbstractOptimizer = NelderMead(),
    optim_kwargs...,
) where {FT <: Real}
    optim_opts_defaults = (; x_tol = 1e-5, f_tol = 1e-5)
    optim_opts = merge(optim_opts_defaults, optim_kwargs)
    optim_opts = Optim.Options(; optim_opts...)

    # Numerically estimate μ_u, σ_u for unbounded distribution which reproduce desired
    # μ_c, σ_c in constrained, transformed coordinates. Unlike other cases, can't be solved
    # for analytically.

    cons = bounded(lower_bound, upper_bound)
    init_μ_u, init_σ_u = _constrained_gaussian_guess(μ_c, σ_c, lower_bound, upper_bound)
    # optimize in log σ to avoid constraint σ>0
    init_logσ_u = log(init_σ_u)

    # Optimize parameters; by default this is done in a quick-and-dirty way, without gradient
    # info (simplex method), since we only optimize 2 parameters.
    # Optimization is finicky since problem becomes singular for large |μ_u|, σ_u: the 
    # distribution becomes collapsed against the bounds of the interval.
    function _optim_fn(μlogσ::Vector{Float64})
        m, s = _mean_std(μlogσ[1], exp(μlogσ[2]), cons)
        return (m - μ_c)^2 + (s - σ_c)^2
    end

    opt = optimize(_optim_fn, [init_μ_u, init_logσ_u], optim_algorithm, optim_opts)
    μ_u = Optim.minimizer(opt)[1]
    σ_u = exp(Optim.minimizer(opt)[2])

    m_c, s_c = _mean_std(μ_u, σ_u, cons)
    if ~isapprox(μ_c, m_c, atol = 1e-3, rtol = 1e-2)
        @warn "Unable to set constrained mean for `$(name)`: target = $(μ_c), got $(m_c)"
    end
    if ~isapprox(σ_c, s_c, atol = 1e-3, rtol = 1e-2)
        @warn "Unable to set constrained std for `$(name)`: target = $(σ_c), got $(s_c)"
    end
    return (μ_u, σ_u)
end

end # of module
