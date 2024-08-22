# included at EOF within ParameterDistributions Module


#usings

#imports - we must import GRF not using or we get annoying warnings for the include of sample into the workspace
import GaussianRandomFields
const GRF = GaussianRandomFields

#exports
#types
export GaussianRandomFieldsPackage
export GRFJL
export GaussianRandomFieldInterface

export sample, build_function_sample
export get_grf
export get_package, spectrum, n_dofs, eval_pts, n_eval_pts, input_dims # for function-based distributions


abstract type FunctionParameterDistributionType <: ParameterDistributionType end

"""
$(DocStringExtensions.TYPEDEF)

Type to dispatch which Gaussian Random Field package to use:

 - `GRFJL` uses the Julia Package [`GaussianRandomFields.jl`](https://github.com/PieterjanRobbe/GaussianRandomFields.jl) 

"""
abstract type GaussianRandomFieldsPackage end
struct GRFJL <: GaussianRandomFieldsPackage end

"""
    struct GaussianRandomFieldInterface <: FunctionParameterDistributionType

GaussianRandomFieldInterface object based on a GRF package. Only a ND->1D output-dimension field interface is implemented.

# Fields

$(TYPEDFIELDS)
"""
struct GaussianRandomFieldInterface <: FunctionParameterDistributionType
    "GRF object, containing the mapping from the field of coefficients to the discrete function"
    gaussian_random_field::Any
    "the choice of GRF package"
    package::GaussianRandomFieldsPackage
    "the distribution of the coefficients that we shall compute with"
    distribution::ParameterDistribution
    function GaussianRandomFieldInterface(
        gaussian_random_field::Any,
        package::GRFJL,
        coefficient_prior::ParameterDistribution,
    )
        n_dof = n_dofs(gaussian_random_field, package)
        ndim = ndims(coefficient_prior)
        if !(ndim == n_dof)
            throw(
                ArgumentError(
                    "The implemented Random Field requires $n_dof coefficients, but the prior distribution provided is only for $ndim coefficients",
                ),
            )
        end
        # create a distribution over which to sample coefficients (degrees of freedom)

        return new(gaussian_random_field, package, coefficient_prior)

    end

end

"""
gets the package type used to construct the GRF
"""
get_package(grfi::GaussianRandomFieldInterface) = grfi.package
"""
gets the distribution, i.e. Gaussian random field object
"""
get_grf(grfi::GaussianRandomFieldInterface) = grfi.gaussian_random_field
"""
gets the, distribution over the coefficients
"""
get_distribution(grfi::GaussianRandomFieldInterface) = grfi.distribution


### Functions to look at grf properties i.e. (related to function view)
"""
    spectrum(grfi::GaussianRandomFieldInterface)

the spectral information of the GRF, e.g. the Karhunen-Loeve coefficients and eigenfunctions if using this decomposition
"""
spectrum(grfi::GaussianRandomFieldInterface) = spectrum(get_grf(grfi), get_package(grfi))

"""
    input_dims(grfi::GaussianRandomFieldInterface

the number of input dimensions of the GRF
"""
input_dims(grfi::GaussianRandomFieldInterface) = input_dims(get_grf(grfi), get_package(grfi))

"""
    eval_pts(grfi::GaussianRandomFieldInterface)

the discrete evaluation point grid, stored as a range in each dimension
"""
eval_pts(grfi::GaussianRandomFieldInterface) = eval_pts(get_grf(grfi), get_package(grfi))

"""
    n_eval_pts(grfi::GaussianRandomFieldInterface)

the number of total discrete evaluation points
"""
n_eval_pts(grfi::GaussianRandomFieldInterface) = n_eval_pts(get_grf(grfi), get_package(grfi))

"""
    n_dofs(grfi::GaussianRandomFieldInterface)

the number of degrees of freedom / coefficients (i.e. the number of parameters)
"""
n_dofs(grfi::GaussianRandomFieldInterface) = n_dofs(get_grf(grfi), get_package(grfi))

"""
    build_function_sample(grfi::GaussianRandomFieldInterface, coeff_vecormat::AbstractVecOrMat, n_draws::Int)

build function `n_draw` times on the discrete grid, given the coefficients `coeff_vecormat`.

Defaults: `n_draw = 1`.
"""
build_function_sample(grfi::GaussianRandomFieldInterface, coeff_vecormat::AbstractVecOrMat, n_draws::Int) =
    build_function_sample(get_grf(grfi), coeff_vecormat, n_draws, get_package(grfi)) # most general case
build_function_sample(grfi::GaussianRandomFieldInterface, coeff_vecormat::AbstractVecOrMat) =
    build_function_sample(grfi, coeff_vecormat, 1) # sample with rng once

"""
    build_function_sample(rng::AbstractRNG, grfi::GaussianRandomFieldInterface, n_draws::Int)

sample function distribution `n_draw` times on the discrete grid, from the stored prior distributions.

Defaults: `n_draw = 1`, `rng = Random.GLOBAL_RNG`, and `coeff_vec` sampled from the stored prior distribution
"""
build_function_sample(rng::AbstractRNG, grfi::GaussianRandomFieldInterface, n_draws::Int) =
    build_function_sample(rng, get_grf(grfi), get_distribution(grfi), n_draws, get_package(grfi)) #most general case
build_function_sample(rng::AbstractRNG, grfi::GaussianRandomFieldInterface) = build_function_sample(rng, grfi, 1) # sample with rng once
build_function_sample(grfi::GaussianRandomFieldInterface) = build_function_sample(Random.GLOBAL_RNG, grfi, 1) #sample with GLOBAL_RNG once
build_function_sample(grfi::GaussianRandomFieldInterface, n_draws::Int) =
    build_function_sample(Random.GLOBAL_RNG, grfi, n_draws) #sample with GLOBAL_RNG, n_draw times

## GRFJL-specific function - including a constructor
spectrum(g, ::GRFJL) = g.data # the spectrum (KL coeffs and eigenfunctions)
eval_pts(g, ::GRFJL) = g.pts
input_dims(g, pkg::GRFJL) = length(eval_pts(g, pkg))
n_eval_pts(g, pkg::GRFJL) = prod(length(eval_pts(g, pkg)[i]) for i in 1:input_dims(g, pkg)) # number of evaluation points
n_dofs(g, ::GRFJL) = GRF.randdim(g) # e.g. the number of terms in the truncated KL expansion

function build_function_sample(
    rng::AbstractRNG,
    g,
    coefficient_distribution::ParameterDistribution,
    n_draws::Int,
    pkg::GRFJL,
)
    # this is effectively sampling from our "prior"
    coeff_mat = sample(rng, coefficient_distribution, n_draws)
    constrained_coeff_mat = transform_unconstrained_to_constrained(coefficient_distribution, coeff_mat)
    return build_function_sample(g, constrained_coeff_mat, n_draws, pkg)
end

function build_function_sample(g, coeff_vecormat::AbstractVecOrMat, n_draws::Int, pkg::GRFJL)

    coeff_mat = isa(coeff_vecormat, AbstractVector) ? repeat(coeff_vecormat, 1, n_draws) : coeff_vecormat

    n_pts = n_eval_pts(g, pkg)
    n_dof = n_dofs(g, pkg)

    if !(size(coeff_mat) == (n_dof, n_draws))
        throw(
            DimensionMismatch(
                "Coefficients provided must be of size ($n_dof, $n_draws) or ($n_dof,), instead received: " *
                string(size(coeff_vecormat)),
            ),
        )
    end

    # now sample a unit normal and multiply by the coefficients
    normal_samples = coeff_mat
    pt_samples = zeros(n_pts, n_draws)
    #now sample with GRF package and flatten (consistent with pts sizes in each dim)
    for i in 1:n_draws
        pt_samples[:, i] = reshape(GRF.sample(g, xi = normal_samples[:, i]), :, 1)
    end

    return pt_samples
end


"""
    GaussianRandomFieldInterface(gaussian_random_field::Any, package::GRFJL)

Constructor of the interface with GRFJL package. Internally this constructs a prior for the degrees of freedom of the function distribution

"""
function GaussianRandomFieldInterface(gaussian_random_field::Any, package::GRFJL)
    n_dof = n_dofs(gaussian_random_field, package)
    # create a distribution over which to sample coefficients (degrees of freedom)
    coefficient_prior = constrained_gaussian("GRF_coefficients", 0.0, 1.0, -Inf, Inf, repeats = n_dof)

    return GaussianRandomFieldInterface(gaussian_random_field, package, coefficient_prior)

end


## Functions to access at coeff (which is a ParameterDistribution)
mean(grfi::GaussianRandomFieldInterface) = mean(get_distribution(grfi))
var(grfi::GaussianRandomFieldInterface) = var(get_distribution(grfi))
cov(grfi::GaussianRandomFieldInterface) = cov(get_distribution(grfi))
get_n_samples(grfi::GaussianRandomFieldInterface) = get_n_samples(get_distribution(grfi))
logpdf(grfi::GaussianRandomFieldInterface, xarray::AbstractVector) = logpdf(get_distribution(grfi), xarray)
sample(grfi::GaussianRandomFieldInterface) = sample(get_distribution(grfi)) #sample with GLOBAL_RNG once
sample(grfi::GaussianRandomFieldInterface, n_draws::Int) = sample(get_distribution(grfi), n_draws) #sample with GLOBAL_RNG, n_draw times
sample(rng::AbstractRNG, grfi::GaussianRandomFieldInterface) = sample(rng, get_distribution(grfi)) # sample with rng once
sample(rng::AbstractRNG, grfi::GaussianRandomFieldInterface, n_draws::Int) =
    sample(rng, get_distribution(grfi), n_draws) #sample with rng, n_draw times


# let's say we have the computational vector (u_p,u_f) for parametric and coeff-of-function parameters
# - apply transform on comp vec (u_p,u_f → ψ_p,ψ_f), here func inherits transform from its coeff_prior.
# - apply build function to ψ_f → ϕ_f evaluated on the discrete grid
# - apply transform on ϕ_f → φ_f to constrain the discrete function
# So in the end we have a map from u = (u_p,u_f) → (ψ_p,ϕ_f)  

# unlike the other PDT, grfi contains internal constraints for the coeffs
"""
    get_all_constraints(grfi::GaussianRandomFieldInterface) = get_all_constraints(get_distribution(grfi))

gets all the constraints of the internally stored coefficient prior distribution of the GRFI
"""
get_all_constraints(grfi::GaussianRandomFieldInterface) = get_all_constraints(get_distribution(grfi))

"""
    ndims(grfi::GaussianRandomFieldInterface, function_parameter_opt = "dof")

Provides a relevant number of dimensions in different circumstances, If `function_parameter_opt` =
- "dof"       : returns `n_dofs(grfi)`, the degrees of freedom in the function
- "eval"      : returns `n_eval_pts(grfi)`, the number of discrete evaluation points of the function
- "constraint": returns `1`, the number of constraints in the evaluation space 
"""
function ndims(grfi::GaussianRandomFieldInterface; function_parameter_opt::AbstractString = "dof")
    if function_parameter_opt == "dof"
        return n_dofs(grfi)
    elseif function_parameter_opt == "eval"
        return n_eval_pts(grfi)
    elseif function_parameter_opt == "constraint"
        return 1
    else
        throw(
            ArgumentError(
                "Keyword options for ndims must be: \"dof\", \"eval\", or \"constraint\". Received $function_parameter_opt ",
            ),
        )
    end
end

# Specific methods for FunctionParameterDistributionType
"""
    transform_unconstrained_to_constrained(d::GaussianRandomFieldInterface, constraint::AbstractVector, x::AbstractVector)

Optional Args build_flag::Bool = true

Two functions, depending on `build_flag`
If `true`, assume x is a vector of coefficients. Perform the following 3 maps. 
1. Apply the transformation to map (possibly constrained) parameter samples `x` into the unconstrained space. Using internally stored constraints (given by the coefficient prior)
2. Build the unconstrained (flattened) function sample at the evaluation points from these constrained coefficients.
3. Apply the constraint from `constraint` to the output space of the function.
If `false`, Assume x is a flattened vector of evaluation points. Apply only step 3. above to x.
"""
function transform_unconstrained_to_constrained(
    d::GaussianRandomFieldInterface,
    constraint::AbstractVector,
    x::AbstractVector{FT};
    build_flag::Bool = true,
) where {FT <: Real}

    if build_flag
        #first transform coeffs using internal constraints
        coeff_constraints = get_all_constraints(d)
        constrained_coeffs =
            cat([c.unconstrained_to_constrained(x[i]) for (i, c) in enumerate(coeff_constraints)]..., dims = 1)
        # then build the discrete function
        function_sample = build_function_sample(d, constrained_coeffs) # n_eval_pts x 1 (vec case)
    else
        function_sample = x
    end

    # then apply the parameter distribution constraint on the output space
    return constraint[1].unconstrained_to_constrained.(function_sample)

end


"""
    transform_unconstrained_to_constrained(d::GaussianRandomFieldInterface, constraint::AbstractVector, x::AbstractMatri)

Optional args: build_flag::Bool = true

Two functions, depending on `build_flag`
If `true`, assume x is a matrix with columns of coefficient samples. Perform the following 3 maps. 
1. Apply the transformation to map (possibly constrained) parameter samples `x` into the unconstrained space. Using internally stored constraints (given by the coefficient prior)
2. Build the unconstrained (flattened) function sample at the evaluation points from these constrained coefficients.
3. Apply the constraint from `constraint` to the output space of the function.
If `false`, Assume x is a matrix with columns as flattened samples of evaluation points. Apply only step 3. above to x.
"""
function transform_unconstrained_to_constrained(
    d::GaussianRandomFieldInterface,
    constraint::AbstractVector,
    x::AbstractMatrix{FT};
    build_flag::Bool = true,
) where {FT <: Real}

    if build_flag
        # first transform coeffs with internal constraints
        coeff_constraints = get_all_constraints(d)
        constrained_coeff =
            Array(hcat([c.unconstrained_to_constrained.(x[i, :]) for (i, c) in enumerate(coeff_constraints)]...)')
        # then build the discrete function
        n_draws = size(constrained_coeff, 2)
        function_samples = build_function_sample(d, constrained_coeff, n_draws) # n_eval_pts x n_coeff_samples
    else
        function_samples = x
    end

    # then apply the parameter distribution constraint on the output space
    return constraint[1].unconstrained_to_constrained.(function_samples)

end

## Note that this is no longer an inverse to the above! Should be noted in the docs.
"""
    transform_constrained_to_unconstrained(d::GaussianRandomFieldInterface, constraint::AbstractVector, x::AbstractMatrix)

Assume x is a matrix with columns as flattened samples of evaluation points.
Remove the constraint from `constraint` to the output space of the function.
Note this is the inverse of `transform_unconstrained_to_constrained(...,build_flag=false)`
"""
function transform_constrained_to_unconstrained(
    d::GaussianRandomFieldInterface,
    constraint::AbstractVector,
    x::AbstractMatrix{FT},
) where {FT <: Real}

    return constraint[1].constrained_to_unconstrained.(x)

end

"""
    transform_constrained_to_unconstrained(d::GaussianRandomFieldInterface, constraint::AbstractVector, x::AbstractVector)

Assume x is a flattened vector of evaluation points.
Remove the constraint from `constraint` to the output space of the function.
Note this is the inverse of `transform_unconstrained_to_constrained(...,build_flag=false)`
"""
function transform_constrained_to_unconstrained(
    d::GaussianRandomFieldInterface,
    constraint::AbstractVector,
    x::AbstractVector{FT},
) where {FT <: Real}

    return constraint[1].constrained_to_unconstrained.(x)

end
