# [Defining prior distributions](@id parameter-distributions)

Bayesian inference begins with an explicit prior distribution. This page describes the interface EnsembleKalmanProcesses provides for specifying priors on parameters, via the ParameterDistributions module (`src/ParameterDistributions.jl`).

## Summary

### ParameterDistribution objects

A prior is specified by a `ParameterDistribution` object, which has three components:

 1. The distribution itself, given as a [ParameterDistributionType](@ref) object. This includes standard Julia [Distributions](https://juliastats.org/Distributions.jl/stable/) as well as empirical/sample-based distributions, and can be uni- or multi-variate. To clarify, despite our use of the term "Kalman processes," the prior distribution is *not* required to be Gaussian.
 2. A constraint (or array of constraints) on the domain of the distribution, given as a [ConstraintType](@ref) or `Array{ConstraintType}` object (the latter case builds a multivariate constraint as the Cartesian product of one-dimensional Constraints). This is used to enforce physical parameter values during inference: the model is never evaluated at parameter values outside the constrained region, and the posterior distribution will only be supported there.
 3. The parameter name, given as a `String`.

In multiparameter settings, one should define one `ParameterDistribution` per parameter, and then concatenate these either in the constructor or with `combine_distributions`. This is illustrated below and in the [Multidimensional example](@ref).

!!! note "What's up with the notation u, ϕ, and θ?"
    Parameters in unconstrained spaces are often denoted ``u`` or ``\theta`` in the literature. In the code, method names featuring `_u` imply the return of a computational, unconstrained parameter.
    
    Parameters in physical/constrained spaces are often denoted ``\mathcal{T}^{-1}(u)``, ``\mathcal{T}^{-1}(\theta)``, or ``\phi`` in the literature (for some bijection ``\mathcal{T}`` mapping to the unbounded space). In the code, method names featuring `_ϕ` imply the return of a physical, constrained parameter, and will always require a `prior` as input to perform the transformations internally.
    
    For more notations see our [Glossary](@ref).

### [Recommended constructor](@id constrained-gaussian)

`constrained_gaussian()` is a streamlined constructor for `ParameterDistribution`s which addresses the most common use case; more general forms of the constructor are documented below, but we **highly** recommend that users begin here when it comes to specifying priors, only using the general constructor when necessary.

Usage:

```julia
prior_1 = constrained_gaussian("param_1", μ_1, σ_1, lower_bound, upper_bound)
prior_2 = constrained_gaussian("param_2", μ_2, σ_2, 0.0, Inf)
prior = combine_distributions([prior_1, prior_2])
```

`prior_1` is a `ParameterDistribution` describing a prior distribution for a parameter named `"param_1"` taking values on the interval [`lower_bound`, `upper_bound`]; the prior distribution has mean μ₁ and standard deviation σ₁.

The use case `constrained_gaussian()` addresses is when prior information is qualitative, and exact distributions of the priors are unknown: i.e., the user is only able to specify the physical and likely ranges of prior parameter values at a rough, qualitative level. `constrained_gaussian()` does this by constructing a `ParameterDistribution` corresponding to a Gaussian "squashed" to fit in the given constraint interval, such that the "squashed" distribution has the specified mean and standard deviation (e.g. `prior_2` above is a log-normal).

The parameters of the Gaussian are chosen automatically (depending on the constraint) to reproduce the desired μ and σ — per the use case, other details of the form of the prior distribution shouldn't be important for downstream inference!

### Recommended constructor - Simple example

Task: We wish to create a prior for a one-dimensional parameter. Our problem dictates that this parameter is bounded between 0 and 1; domain knowledge leads us to expect it should be around 0.7. The parameter is called `point_seven`.

We're told that the prior mean is 0.7; we choose a prior standard deviation of 0.15 to be sufficiently wide without putting too much mass at the upper bound. The constructor is then
```julia
prior = constrained_gaussian("point_seven", 0.7, 0.15, 0.0, 1.0)
```

```@setup example_one
# instead of importing ParameterDistributions & dependencies to call constructor,
# which would make docs build longer and more fragile, simply hard-code Normal()
# parameters found by constrained_gaussian constructor

using Distributions
using Plots
Plots.default(lw=2)

N = 50
x_eval = collect(-5:6/400:5)

#bounded in [0.0, 1.0]
transform_unconstrained_to_constrained(x) = 1.0 - 1.0 / (exp(x) + 1)
dist= pdf.(Normal(0.957711, 0.78507), x_eval)
constrained_x_eval = transform_unconstrained_to_constrained.(x_eval)

p2 = plot(constrained_x_eval, dist) 
vline!([0.7]) 
title!("Prior pdf")
```

The pdf of the constructed prior distribution (in the physical, constrained space) looks like:
```@example example_one
p = plot(p2, legend=false, size = (450, 450)) #hide
```

In [Simple example revisted](@ref) below, we repeat this example "manually" with the general constructor.

!!! note "What if I want to impose the same prior on many parameters?"
    
    The recommended constructor can be called as `constrained_gaussian(...; repeats = n)` to return a combined prior formed by `n` identical priors.


## ParameterDistribution class

This section provides more details on the components of a `ParameterDistribution` object.

### ParameterDistributionType

The `ParameterDistributionType` class comprises three subclasses for specifying different types of prior distributions:

 - The `Parameterized` type is initialized using a Julia `Distributions.jl` object. Samples are drawn randomly from the distribution object.
 - The `VectorOfParameterized` type is initialized with a vector of distributions.
 - The `Samples` type is initialized using a two dimensional array. Samples are drawn randomly (with replacement) from the columns of the provided array.

!!! warning
    We recommend that the distributions be unbounded (see next section), as the filtering algorithms in EnsembleKalmanProcesses are not guaranteed to preserve constraints unless defined through the `ConstraintType` mechanism.


### ConstraintType

The inference algorithms implemented in EnsembleKalmanProcesses assume unbounded parameter domains. To be able to handle constrained parameter values consistently, the ConstraintType class defines a bijection between the physical, constrained parameter domain and an unphysical, unconstrained domain in which the filtering takes place. This bijection is specified by the functions `transform_constrained_to_unconstrained` and `transform_unconstrained_to_constrained`, which are built from either predefined constructors or user-defined constraint functions given as arguments to the `ConstraintType` constructor. 

!!! warning
    When a nontrivial `ConstraintType` is given, the general constructor assumes the `ParameterDistributionType` is specified in the *unconstrained* space; the actual prior pdf is then the composition of the `ParameterDistributionType`'s pdf with the `transform_unconstrained_to_constrained` transformation. We provide `constrained_gaussian` to define priors directly in the physical, constrained space.

!!! warning
    It is up to the user to ensure any custom mappings `transform_constrained_to_unconstrained` and `transform_unconstrained_to_constrained` are inverses of each other.

We provide the following predefined constructors which implement mappings that handle the most common constraints:

 - `no_constraint()`: The parameter is unconstrained and takes values in (-∞, ∞) (mapping is the identity).
 - `bounded_below(lower_bound)`: The parameter takes values in [`lower_bound`, ∞).
 - `bounded_above(upper_bound)`: The parameter takes values in (-∞, `upper_bound`].
 - `bounded(lower_bound,upper_bound)`: The parameter takes values on the interval [`lower_bound`, `upper_bound`].

 These are demonstrated in [ConstraintType Examples](@ref).

Currently we only support multivariate constraints which are the Cartesian product of the one-dimensional `ConstraintType`s. Every component of a multidimensional parameter must have an associated constraint, so, e.g. for a multivariate `ParameterDistributionType` of dimension `p` the user must provide a `p-`dimensional `Array{ConstraintType}`. A `VectorOfParameterized` distribution built with distributions of dimension `p` and `q` has dimension `p+q`.

### The name

This is simply a `String` used to identify different parameters in multi-parameter situations, as in the methods below.


## ParameterDistribution constructor

The [Recommended constructor](@ref constrained-gaussian), `constrained_gaussian()`, is described above. For more general cases in which the prior needs to be specified in more detail, a `ParameterDistribution` may be constructed "manually" from its component objects:

```julia
prior_1 = ParameterDistribution(distribution_1, constraint_1, name_1)
prior_2 = ParameterDistribution(distribution_2, constraint_2, name_2)
prior = combine_distributions( [prior_1, prior_2])
```

Arguments may also be provided as a `Dict`:
```julia
dict_1 = Dict("distribution" => distribution_1, "constraint" => constraint_1, "name" => name_1)
dict_2 = Dict("distribution" => distribution_2, "constraint" => constraint_2, "name" => name_2)
prior = ParameterDistribution( [dict_1, dict_2] )
```

We provide [Additional Examples](@ref) below; see also examples in the package `examples/` and unit tests found in `test/ParameterDistributions/runtests.jl`.


## ParameterDistribution methods

These functions typically return a `Dict` with `ParameterDistribution.name` as a keys, or an `Array` if requested:

 - `get_name`: returns the name(s) of parameters in the `ParameterDistribution`.
 - `get_distribution`: returns the distributions (`ParameterDistributionType` objects) in the `ParameterDistribution`. Note that this is *not* the prior pdf used for inference if nontrivial constraints have been applied.
 - `mean, var, cov, sample, logpdf`: mean, variance, covariance, logpdf or samples the Julia Distribution if `Parameterized`, or draws from the list of samples if `Samples`. Extends the StatsBase definitions. Note that these do *not* correspond to the prior pdf used for inference if nontrivial constraints have been applied.
 - `transform_unconstrained_to_constrained`: Applies the constraint mappings.
 - `transform_constrained_to_unconstrained`: Applies the inverse constraint mappings.



## Additional Examples

### Simple example revisted

To illustrate what the `constrained_gaussian` constructor is doing, in this section we repeat the [Recommended constructor - Simple example](@ref) given above, using the "manual," general-purpose constructor.

Let's initialize the constraint first,
```julia
constraint = bounded(0, 1)
```
This defines the following transformation to the  constrained space (and also its inverse)
```julia
transform_unconstrained_to_constrained(x) = exp(x) / (exp(x) + 1)
```
The prior mean should be around 0.7 (in the constrained space), and one can find that the push-forward of a particular normal distribution, namely, `transform_unconstrained_to_constrained(Normal(mean = 1, sd = 0.5))` gives a prior pdf with 95% of its mass between [0.5, 0.88]. 

This is the main difference from the use of the `constrained_gaussian` constructor: in that example, the constructor numerically solved for the parameters of the Normal() which would reproduce the requested μ, σ for the physical, constrained quantity (since no closed-form transformation for the moments exists.)

```julia
distribution = Parameterized(Normal(1, 0.5))
```
Finally we attach the name
```julia
name = "point_seven"
```
and the distribution is created by either:
```julia
prior = ParameterDistribution(distribution, constraint, name)
```
or
```julia
prior_dict = Dict("distribution" => distribution, "constraint" => constraint, "name" => name)
prior = ParameterDistribution(prior_dict)
```

```@setup example_two
using Distributions
using Plots
Plots.default(lw=2)

N = 50
x_eval = collect(-3:6/400:3)

#bounded in [0.0, 1.0]
transform_unconstrained_to_constrained(x) = exp(x) / (exp(x) + 1)
dist= pdf.(Normal(1, 0.5), x_eval)
constrained_x_eval = transform_unconstrained_to_constrained.(x_eval)

p1 = plot(x_eval, dist,) 
vline!([1.0]) 
title!("Normal(1, 0.5)")

p2 = plot(constrained_x_eval, dist) 
vline!([transform_unconstrained_to_constrained(1.0)]) 
title!("Constrained Normal(1, 0.5)")
```

The pdf of the Normal distribution and its transform to the physical, constrained space are:
```@example example_two
p = plot(p1, p2, legend=false, size = (900, 450)) #hide
```

### Multidimensional example

To show how to combine priors in a more realistic setting (e.g. for an entire parametrized process), we create a 25-dimensional parameter distribution from three dictionaries.

The first parameter is a 3-dimensional distribution, with the following bound constraints on parameters in physical space:
```julia
c1 = [bounded_below(0.0),
      bounded_below(0.0),
      bounded_below(0.0)]  
```
We know that a multivariate normal represents its distribution in the transformed (unbounded) space. Here we take a tridiagonal covariance matrix.
```julia
diag_val = 0.5 * ones(3)
udiag_val = 0.25 * ones(1)
mean = ones(4)
covariance = SymTridiagonal(diagonal_val, udiag_val)
d1 = Parameterized(MvNormal(mean, covariance)) # 4D multivariate normal
```
We also provide a name
```julia
name1 = "constrained_mvnormal"
```

The second parameter is a 2-dimensional one. It is only given by 4 samples in the transformed space - (where one will typically generate samples). It is bounded in the first dimension by the constraint shown, there is a user provided transform for the second dimension - using the default constructor.

```julia
d2 = Samples([1.0 3.0; 5.0 7.0; 9.0 11.0; 13.0 15.0]) # 4 samples of 2D parameter space
transform = (x -> 3 * x + 14)
inverse_transform = (x -> (x - 14) / 3)
c2 = [bounded(10, 15),
      Constraint(transform, inverse_transform)]
name2 = "constrained_sampled"
```

The final parameter is 20-dimensional, defined as a list of i.i.d univariate distributions we make use of the `VectorOfParameterized` type
```julia
d3 = VectorOfParameterized(repeat([Beta(2,2)],20))
c3 = repeat([no_constraint()],20)
name3 = "Beta"
```

The full prior distribution for this setting is created either through building simple distributions and combining
```julia
u1 = ParameterDistribution(d1, c1, name1)
u2 = ParameterDistribution(d2, c2, name2)
u3 = ParameterDistribution(d3, c3, name3)
u = combine_distributions( [u1, u2, u3])
```

or an array of the parameter specifications as dictionaries.
```julia
param_dict1 = Dict("distribution" => d1, "constraint" => c1, "name" => name1)
param_dict2 = Dict("distribution" => d2, "constraint" => c2, "name" => name2)
param_dict3 = Dict("distribution" => d3, "constraint" => c3, "name" => name3)
u = ParameterDistribution([param_dict1, param_dict2, param_dict3])
```


## ConstraintType Examples

For each for the predefined `ConstraintType`s, we present animations of the resulting constrained prior distribution for
```julia
distribution = Parameterized(Normal(μ, σ))
```
where we vary μ and σ respectively. As noted above, in the presence of a nontrivial constraint, μ and σ will no longer correspond to the mean and standard deviation of the prior distribution (which is taken in the physical, constrained space).

### Without constraints: `"constraint" => no_constraints()`

The following specifies a prior based on an unconstrained `Normal(0.5, 1)` distribution:

```julia
using EnsembleKalmanProcesses.ParameterDistributions
using Distributions 

param_dict = Dict(
"distribution" => Parameterized(Normal(0.5, 1)),
"constraint" => no_constraint(),
"name" => "unbounded_parameter",
)

prior = ParameterDistribution(param_dict)
```
where `no_constraint()` automatically defines the identity constraint map
```julia
transform_unconstrained_to_constrained(x) = x
```
The following plots show the effect of varying μ and σ in the constrained space (which is trivial here):

```@setup no_constraints
using Distributions
using Plots
Plots.default(lw=2)

N = 50
x_eval = collect(-5:10/200:5)
mean_varying = collect(-3:6/(N+1):3)
sd_varying = collect(0.1:2.9/(N+1):3)

# no constraint
transform_unconstrained_to_constrained(x) = x

mean0norm(n) = pdf.(Normal(0, sd_varying[n]), x_eval)
sd1norm(n) = pdf.(Normal(mean_varying[n], 1), x_eval)
constrained_x_eval = transform_unconstrained_to_constrained.(x_eval)

p1 = plot(constrained_x_eval, mean0norm.(1))
vline!([transform_unconstrained_to_constrained(0)])

p2 = plot(constrained_x_eval, sd1norm.(1))
vline!([transform_unconstrained_to_constrained(mean_varying[1])])

p = plot(p1, p2, layout=(1, 2), size = (900, 450), legend = false)
 
anim_unbounded = @animate for n = 1:length(mean_varying)
   #set new y data 
   p[1][1][:y] = mean0norm(n) 
   p[1][:title] = "Transformed Normal(0, " * string(round(sd_varying[n], digits=3)) * ")" 
   p[2][1][:y] = sd1norm(n) 
   p[2][2][:x] = [transform_unconstrained_to_constrained(mean_varying[n]),
                  transform_unconstrained_to_constrained(mean_varying[n]),
                  transform_unconstrained_to_constrained(mean_varying[n])]

   p[2][:title] = "Transformed Normal(" * string(round(mean_varying[n], digits=3)) * ", 1)"
end 
```

```@example no_constraints
gif(anim_unbounded, "anim_unbounded.gif", fps = 5) # hide
```



### Bounded below by 0: `"constraint" => bounded_below(0)`

The following specifies a prior for a parameter which is bounded below by 0 (i.e. its only physical values are positive), and which has a `Normal(0.5, 1)` distribution in the unconstrained space:

```julia
using EnsembleKalmanProcesses.ParameterDistributions
using Distributions 

param_dict = Dict(
"distribution" => Parameterized(Normal(0.5, 1)),
"constraint" = bounded_below(0),
"name" => "bounded_below_parameter",
)

prior = ParameterDistribution(param_dict)
```
where `bounded_below(0)` automatically defines the constraint map
```julia
transform_unconstrained_to_constrained(x) = exp(x)
```
The following plots show the effect of varying μ and σ in the physical, constrained space:

```@setup bounded_below
using Distributions
using Plots
Plots.default(lw=2)

N = 50
x_eval = collect(-5:10/400:5)
mean_varying = collect(-1:5/(N+1):4)
sd_varying = collect(0.1:3.9/(N+1):4)

#bounded below by 0
transform_unconstrained_to_constrained(x) = exp(x)

mean0norm(n) = pdf.(Normal(0,sd_varying[n]), x_eval)
sd1norm(n) = pdf.(Normal(mean_varying[n], 1), x_eval)
constrained_x_eval = transform_unconstrained_to_constrained.(x_eval)

p1 = plot(constrained_x_eval, mean0norm.(1))
vline!([transform_unconstrained_to_constrained(0)])

p2 = plot(constrained_x_eval, sd1norm.(1))
vline!([transform_unconstrained_to_constrained(mean_varying[1])])

p = plot(p1,p2, layout=(1,2), size = (900,450), legend=false)
 
anim_bounded_below = @animate for n = 1:length(mean_varying) 
   #set new y data  
   p[1][1][:y] = mean0norm(n) 
   p[1][:title] = "Transformed Normal(0, " * string(round(sd_varying[n], digits=3)) * ")"
   p[2][1][:y] = sd1norm(n) 
   p[2][2][:x] = [transform_unconstrained_to_constrained(mean_varying[n]),
                  transform_unconstrained_to_constrained(mean_varying[n]),
                  transform_unconstrained_to_constrained(mean_varying[n])]
   p[2][:title] = "Transformed Normal(" * string(round(mean_varying[n], digits=3)) * ", 1)"
end 
```

```@example bounded_below
gif(anim_bounded_below, "anim_bounded_below.gif", fps = 5) # hide
```


### Bounded above by 10.0: `"constraint" => bounded_above(10)`

The following specifies a prior for a parameter which is bounded above by ten, and which has a `Normal(0.5, 1)` distribution in the unconstrained space:
```julia
using EnsembleKalmanProcesses.ParameterDistributions
using Distributions

param_dict = Dict(
"distribution" => Parameterized(Normal(0.5, 1)),
"constraint" => bounded_above(10),
"name" => "bounded_above_parameter",
)
prior = ParameterDistribution(param_dict)
```
where `bounded_above(10)` automatically defines the constraint map
```julia
transform_unconstrained_to_constrained(x) = 10 - exp(-x)
```
The following plots show the effect of varying μ and σ in the physical, constrained space:

```@setup bounded_above
using Distributions
using Plots
Plots.default(lw=2)

N = 50
x_eval = collect(-5:4/400:5)
mean_varying = collect(-1:5/(N+1):4)
sd_varying = collect(0.1:3.9/(N+1):4)

#bounded above by 10.0
transform_unconstrained_to_constrained(x) = 10 - exp(-x)

mean0norm(n) = pdf.(Normal(0, sd_varying[n]), x_eval)
sd1norm(n) = pdf.(Normal(mean_varying[n], 1), x_eval)
constrained_x_eval = transform_unconstrained_to_constrained.(x_eval)

p1 = plot(constrained_x_eval, mean0norm.(1))
vline!([transform_unconstrained_to_constrained(0)])

p2 = plot(constrained_x_eval, sd1norm.(1))
vline!([transform_unconstrained_to_constrained(mean_varying[1])])

p = plot(p1, p2, layout=(1, 2), size = (900, 450), legend=false)
 
anim_bounded_above = @animate for n = 1:length(mean_varying)[1]
  #set new y data
   p[1][1][:y] = mean0norm(n)
   p[1][:title] = "Transformed Normal(0, " * string(round(sd_varying[n], digits=3)) * ")"
   p[2][1][:y] = sd1norm(n)
   p[2][2][:x] = [transform_unconstrained_to_constrained(mean_varying[n]),
                  transform_unconstrained_to_constrained(mean_varying[n]),
                  transform_unconstrained_to_constrained(mean_varying[n])]
   p[2][:title] = "Transformed Normal(" * string(round(mean_varying[n], digits=3)) * ", 1)"
end 
```

```@example bounded_above
gif(anim_bounded_above, "anim_bounded_above.gif", fps = 5) # hide
```


### Bounded between 5 and 10: `"constraint" => bounded(5, 10)`
The following specifies a prior for a parameter whose physical values lie in the range between 5 and 10, and which has a `Normal(0.5, 1)` distribution in the unconstrained space:
```julia
using EnsembleKalmanProcesses.ParameterDistributions
using Distributions

param_dict = Dict(
"distribution" => Parameterized(Normal(0.5, 1)),
"constraint" => bounded(5, 10),
"name" => "bounded_parameter",
)

prior = ParameterDistribution(param_dict)
```
where `bounded(-1, 5)` automatically defines the constraint map
```julia
transform_unconstrained_to_constrained(x) = 10 - 5 / (exp(x) + 1)
```
The following plots show the effect of varying μ and σ in the physical, constrained space:

```@setup bounded
using Distributions
using Plots
Plots.default(lw=2)

N = 50
x_eval = collect(-10:20/400:10)
mean_varying = collect(-3:2/(N+1):3)
sd_varying = collect(0.1:0.9/(N+1):10)

#bounded in [5.0, 10.0]
transform_unconstrained_to_constrained(x) = 10 - 5 / (exp(x) + 1)

mean0norm(n) = pdf.(Normal(0, sd_varying[n]), x_eval)
sd1norm(n) = pdf.(Normal(mean_varying[n], 1), x_eval)
constrained_x_eval = transform_unconstrained_to_constrained.(x_eval)

p1 = plot(constrained_x_eval, mean0norm.(1))
vline!([transform_unconstrained_to_constrained(0)])

p2 = plot(constrained_x_eval, sd1norm.(1))
vline!([transform_unconstrained_to_constrained(mean_varying[1])])

p = plot(p1, p2, layout=(1, 2), size = (900, 450), legend=false)
 
anim_bounded = @animate for n = 1:length(mean_varying)
   #set new y data
   p[1][1][:y] = mean0norm(n)
   p[1][:title] = "Transformed Normal(0, " * string(round(sd_varying[n], digits=3)) * ")"
   p[2][1][:y] = sd1norm(n)
   p[2][2][:x] = [transform_unconstrained_to_constrained(mean_varying[n]),
                  transform_unconstrained_to_constrained(mean_varying[n]),
                  transform_unconstrained_to_constrained(mean_varying[n])]
   p[2][:title] = "Transformed Normal(" * string(round(mean_varying[n], digits=3)) * ", 1)"
end 
```

```@example bounded
gif(anim_bounded, "anim_bounded.gif", fps = 10) # hide
```
