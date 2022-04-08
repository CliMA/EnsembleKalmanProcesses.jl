# Prior distributions

We provide a flexible setup for storing prior distribution with the `ParameterDistributions` module found in `src/ParameterDistributions.jl`.

One can create a full parameter distribution using three inputs:
 1. A distribution, given as a `ParameterDistributionType` object,
 2. A constraint, or array of constraints, given as a `ConstraintType` or `Array{ConstraintType}` object,
 3. A name, given as a `String`.

In complex settings, one should build a `ParameterDistribution`-per-distribution, and then combine together. 

We provide two similar forms of constructor:
```julia
prior_1 = ParameterDistribution(distribution_1, constraint_1, name_1)
prior_2 = ParameterDistribution(distribution_2, constraint_2, name_2)
prior = combine_distributions( [prior_1, prior_2])
```

One can also use the `Dict` based constructor
```julia
dict_1 = Dict("distribution" => distribution_1, "constraint" => constraint_1, "name" => name_1)
dict_2 = Dict("distribution" => distribution_2, "constraint" => constraint_2, "name" => name_2)
prior = ParameterDistribution( [dict_1, dict_2] )
```

We provide many examples in the package, and the unit tests found in `test/ParameterDistributions/runtests.jl`.

# A simple example:
Task: We wish to create a prior for a one-dimensional parameter. Our problem dictates that this parameter is bounded between 0 and 1. Prior knowledge dictates it is around 0.7. The parameter is called `point_seven`.

Solution: We should use a Normal distribution with the predefined "bounded" constraint.

Let's initialize the constraint first,
```julia
constraint = bounded(0, 1)
```
This **automatically** sets up the following transformation to the  constrained space (and also its inverse)
```julia
transform_unconstrained_to_constrained(x) = exp(x) / (exp(x) + 1)
```
The prior should be around 0.7 (in the constrained space), and one can find that the push-forward of a particular normal distribution, namely, `transform_unconstrained_to_constrained(Normal(mean = 1, sd = 0.5))` gives a prior pdf with 95% of its mass between [0.5, 0.88].
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

```@setup zero_point_seven
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
title!("Transformed Normal(1, 0.5)")
```

The pdf of the Normal distribution and its transform look like:
```@example zero_point_seven
p = plot(p1, p2, legend=false, size = (900, 450)) #hide
```

## 1. The ParameterDistributionType

The `ParameterDistributionType` has two flavours for building a distribution:
 - The `Parameterized` type is initialized using a Julia `Distributions.jl` object. Samples are drawn randomly from the distribution object
 - The `Samples` type is initialized using a two dimensional array. Samples are drawn randomly (with replacement) from the columns of the provided array.

One can use combinations of these distributions to construct a full parameter distribution.

!!! note
    We recommend these distributions be unbounded (see about constraints below), as our methods do not preserve constraints directly.

## 2. The ConstraintType

Our implemented algorithms do not work in constrained parameter space directly. Therefore, constraints are tackled by the mappings
`transform_constrained_to_unconstrained` and `transform_unconstrained_to_constrained`. The mappings are built from either predefined or user-defined constraint functions held in the `ConstraintType`. 

In this section we call parameters are one-dimensional. Every parameter must have an associated independent `ConstraintType`, therefore we for each `ParameterDistributionType` of dimension `p` the user must provide a `p-`dimensional `Array{ConstraintType}`.

### Predefined ConstraintTypes

We provide some `ConstraintType`s, which apply different transformations internally to enforce bounds on physical parameter spaces. The types have the following constructors:

 - `no_constraint()`, no transform is required for this parameter,
 - `bounded_below(lower_bound)`, the physical parameter has a (provided) lower bound,
 - `bounded_above(upper_bound)`, the physical parameter has a (provided) upper bound,
 - `bounded(lower_bound,upper_bound)`, the physical parameter has the (provided) bounds.

Users can also define their own transformations by directly creating a `ConstraintType` object with their own mappings.

!!! warn
    It is up to the user to ensure any custom mappings `transform_constrained_to_unconstrained` and `transform_unconstrained_to_constrained` are inverses of each other.

## 3. The name

This is simply a `String` identifier for the parameters later on.

## 4. The power of the normal distribution

The combination of different normal distributions with these predefined constraints, provides a surprising breadth of prior distributions.

!!! note
    We **highly** recommend users start with Normal distribution and predefined `ConstraintType`: these offer the best computational benefits and clearest interpretation of the methods.

For each for the predefined `ConstraintType`s, we present animations of the resulting prior distribution for
```julia
distribution = Parameterized(Normal(mean, sd))
```
where: 
1. we fix the mean value to 0, and range over the standard deviation,
2. we fix the standard deviation to 1 and range over the mean value.

### Without constraints: `"constraint" => no_constraints()`

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

The following generates the transformed `Normal(0.5, 1)` distribution

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

### Bounded below by 0: `"constraint" => bounded_below(0)`

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

The following generates the transformed `Normal(0.5, 1)` distribution

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

### Bounded above by 10.0: `"constraint" => bounded_above(10)`

```@setup bounded_above
using Distributions
using Plots
Plots.default(lw=2)

N = 50
x_eval = collect(-5:4/400:5)
mean_varying = collect(-1:5/(N+1):4)
sd_varying = collect(0.1:3.9/(N+1):4)

#bounded above by 10.0
transform_unconstrained_to_constrained(x) = 10 - exp(x)

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

The following generates the transformed `Normal(0.5, 1)` distribution

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
transform_unconstrained_to_constrained(x) = 10 - exp(x)
```

### Bounded in  between 5 and 10: `"constraint" => bounded(5, 10)`

```@setup bounded
using Distributions
using Plots
Plots.default(lw=2)

N = 50
x_eval = collect(-10:20/400:10)
mean_varying = collect(-3:2/(N+1):3)
sd_varying = collect(0.1:0.9/(N+1):10)

#bounded in [5.0, 10.0]
transform_unconstrained_to_constrained(x) = (10 * exp(x) + 5) / (exp(x) + 1)

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

The following generates the transformed `Normal(0.5, 1)` distribution

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
where `bounded(5, 10)` automatically defines the constraint map
```julia
transform_unconstrained_to_constrained(x) = (10 * exp(x) + 5) / (exp(x) + 1)
```

## A more involved example:

We create a 6-dimensional parameter distribution from two dictionaries.

The first parameter is a 4-dimensional distribution with the following constraints on parameters in physical space:
```julia
c1 = [no_constraint(),     # no constraints
      bounded_below(-1.0), # provide lower bound
      bounded_above(0.4),  # provide upper bound
      bounded(-0.1, 0.2)]  # provide lower and upper bound
```
We choose to use a multivariate normal to represent its distribution in the transformed (unbounded) space. Here we take a tridiagonal covariance matrix.
```julia
diag_val = 0.5 * ones(4)
udiag_val = 0.25 * ones(3)
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
The full prior distribution for this setting is created either through building simple distributions and combining
```julia
u1 = ParameterDistribution(d1, c1, name1)
u2 = ParameterDistribution(d2, c2, name2)
u = combine_distributions( [prior_1, prior_2])
```
or an array of the parameter specifications as dictionaries.
```julia
param_dict1 = Dict("distribution" => d1, "constraint" => c1, "name" => name1)
param_dict2 = Dict("distribution" => d2, "constraint" => c2, "name" => name2)
u = ParameterDistribution([param_dict1, param_dict2])
```

## Other functions
These functions typically return a `Dict` with `ParameterDistribution.name` as a keys, or an `Array` if requested:
 - `get_name`: returns the names
 - `get_distribution`: returns the Julia Distribution object if it is `Parameterized`
 - `mean, var, cov, sample, logpdf`: mean,variance,covariance,logpdf or samples the Julia Distribution if `Parameterized`, or draws from the list of samples if `Samples` extends the StatsBase definitions
 - `transform_unconstrained_to_constrained`: apply the constraint mappings
 - `transform_constrained_to_unconstrained`: apply the inverse constraint mappings
