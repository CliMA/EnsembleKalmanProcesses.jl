# ParameterDistributions

```@meta
CurrentModule = EnsembleKalmanProcesses.ParameterDistributions
```

## ParameterDistributionTypes
```@docs
ParameterDistributionType
Parameterized
Samples
VectorOfParameterized
```

## Constraints
```@docs
ConstraintType
NoConstraint
BoundedBelow
BoundedAbove
Bounded
Constraint
no_constraint
bounded_below
bounded_above
bounded
length(c::CType) where {CType <: ConstraintType}
size(c::CType) where {CType <: ConstraintType}
```

## ParameterDistributions

```@docs
ParameterDistribution
constrained_gaussian
combine_distributions
n_samples
get_name
get_dimensions
get_n_samples
get_all_constraints(::ParameterDistribution)
get_constraint_type
get_bounds
batch
get_distribution
sample
logpdf
mean
var
cov
ndims(pd::ParameterDistribution)
transform_constrained_to_unconstrained(pd::ParameterDistribution, x::AbstractVecOrMat{T}) where {T <: Real}
transform_constrained_to_unconstrained(pd::ParameterDistribution, x::Dict)
transform_constrained_to_unconstrained(pd::ParameterDistribution, x)
transform_unconstrained_to_constrained(pd::ParameterDistribution, x::AbstractVecOrMat{T}) where {T <: Real}
transform_unconstrained_to_constrained(pd::ParameterDistribution, x::Dict)
transform_unconstrained_to_constrained(pd::ParameterDistribution, x)
```

## FunctionParameterDistributions

```@docs
FunctionParameterDistributionType
GaussianRandomFieldsPackage
GRFJL
GaussianRandomFieldInterface
ndims(grfi::GaussianRandomFieldInterface)
get_all_constraints(grfi::GaussianRandomFieldInterface)
transform_constrained_to_unconstrained(::GaussianRandomFieldInterface, ::AbstractVector, ::AbstractVector{FT}) where {FT <: Real}
transform_constrained_to_unconstrained(::GaussianRandomFieldInterface, ::AbstractVector, ::AbstractMatrix{FT}) where {FT <: Real}
transform_unconstrained_to_constrained(::GaussianRandomFieldInterface, ::AbstractVector, ::AbstractVector{FT}) where {FT <: Real}
transform_unconstrained_to_constrained(::GaussianRandomFieldInterface, ::AbstractVector, ::AbstractMatrix{FT}) where {FT <: Real}
get_grf
build_function_sample
get_package
spectrum
n_dofs
eval_pts
n_eval_pts
input_dims
```
