# Prior distributions

We provide a flexible setup for storing prior distribution with the `ParameterDistributionStorage` module found in `src/ParameterDistribution.jl` 

One can create a full parameter distribution using three inputs.
 1. A Distribution, given as a `ParameterDistributionType` object
 2. An array of Constraints, given as a `Array{ConstraintType}` object
 3. A Name, given as a `String` 

One can also provide arrays of the triple (1.,2.,3.) to create more complex distributions

# A simple example:
Task: We wish to create a prior for a one-dimensional parameter. Our problem dictates that this parameter is bounded between 0 and 1. Prior knowledge dictates it is around 0.7. The parameter is called "point_seven".

Solution: We should use a Normal distribution with the predefined "bounded" constraint.

Let's initialize the constraint first,
```julia
constraint = [bounded(0,1)] # Sets up a logit-transformation into [0,1].
```
The prior is around 0.7, and the push forward of a normal distribution N(mean=1,sd=0.5) gives a prior with 95% of it's mass between [0.5,0.88].
```julia
distribution = Parameterized(Normal(1,0.5)) 
```
Finally we attach the name
```julia
name = "point_seven"
```
And the distribution is created by calling:
```julia
prior = ParameterDistribution(distribution,constraint,name)
```

## 1. The ParameterDistributionType

The `ParameterDistributionType` has 2 flavours for building a distribution.
 - The `Parameterized` type is initialized using a Julia `Distributions.jl` object. Samples are drawn randomly from the distribution object
 - The `Samples` type is initialized using a two dimensional array. Samples are drawn randomly (with replacement) from the columns of the provided array

One can use combinations of these distributions to construct a full parameter distribution.

!!! note
    We recommend these distributions be unbounded (see about constraints below), as our methods do not preserve constraints directly.

## 2. The ConstraintType

Our implemented algorithms do not work in constrained parameter space directly. Therefore, constraints are tackled by the mappings
`transform_constrained_to_unconstrained` and `transform_unconstrained_to_constrained`. The mappings are built from either predefined or user-defined constraint functions held in the `ConstraintType`. 

In this section we call parameters are one-dimensional. Every parameter must have an associated independent `ConstraintType`, therefore we for each `ParameterDistributionType` of dimension `p` the user must provide a `p-`dimensional `Array{ConstraintType}`.

### Predefined ConstraintTypes

We provide some ConstraintTypes, which apply different transformations internally to enforce bounds on physical parameter spaces. The types have the following constructors

 - `no_constraint()`, no transform is required for this parameter
 - `bounded_below(lower_bound)`, the physical parameter has a (provided) lower bound
 - `bounded_above(upper_bound)`, the physical parameter has a (provided) upper bound 
 - `bounded(lower_bound,upper_bound)`, the physical parameter has the (provided) bounds

Users can also define their own transformations by directly creating a `ConstraintType` object with their own mappings.

!!! note
    It is up to the user to ensure their provided transforms are inverses of each other.


## 3. The name

This is simply an identifier for the parameters later on.

## A more involved example:

We create a 6-dimensional parameter distribution from 2 triples.

The first triple is a 4-dimensional distribution with the following constraints on parameters in physical space:
```julia
c1 = [no_constraint(), # no constraints
      bounded_below(-1.0), # provide lower bound
      bounded_above(0.4), # provide upper bound
      bounded(-0.1,0.2)] # provide lower and upper bound
```
We choose to use a multivariate normal to represent its distribution in the transformed (unbounded) space. Here we take a tridiagonal covariance matrix.
```julia
diag_val = 0.5*ones(4)
udiag_val = 0.25*ones(3)
mean = ones(4)
covariance = SymTridiagonal(diagonal_val, udiag_val)
d1 = Parameterized(MvNormal(mean,covariance)) # 4D multivariate normal
```
We also provide a name
```julia
name1 = "constrained_mvnormal"
```

The second triple is a 2-dimensional one. It is only given by 4 samples in the transformed space - (where one will typically generate samples). It is bounded in the first dimension by the constraint shown, there is a user provided transform for the second dimension - using the default constructor.

```julia
d2 = Samples([1.0 3.0; 5.0 7.0; 9.0 11.0; 13.0 15.0]) # 4 samples of 2D parameter space
transform = (x -> 3*x + 14)
inverse_transform = (x -> (x-14) / 3)
c2 = [bounded(10,15),
      Constraint(transform, inverse_transform)]
name2 = "constrained_sampled"
```
The full prior distribution for this setting is created with arrays of our two triples

```julia
u = ParameterDistribution([d1,d2],[c1,c2],[name1,name2])
```
## Other functions
These functions typically return a `Dict` with `ParameterDistribution.name` as a keys, or an `Array` if requested 
 - `get_name`: returns the names
 - `get_distribution`: returns the Julia Distribution object if it is `Parameterized`
 - `sample_distribution`: samples the Julia Distribution if `Parameterized`, or draws from the list of samples if `Samples`

 - `transform_unconstrained_to_constrained`: Apply the constraint mappings
 - `transform_constrained_to_unconstrained`: Apply the inverse constraint mappings 
