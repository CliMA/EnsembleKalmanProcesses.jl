# Observations

The `Observations` object is used to store the truth for convenience of the user. The ingredients are
1. Samples of the data `Vector{Vector{Float}}` or `Array{Float, 2}`. If provided as a 2D array, the samples must be provided as *columns*. They are stored internally as `Vector{Vector{Float}}`
2. An optional covariance matrix can be provided.
3. The names of the data in this object as a `String` or `Vector{String}`

The empirical mean is calculated automatically.
If a covariance matrix is not provided, then the empirical covariance is also calculated automatically.

## A simple example:

Here is a typical construction of the object:
```julia
μ = zeros(5)
Γy = rand(5, 5)
Γy = Γy' * Γy
yt = rand(MvNormal(μ, Γy), 100) # generate 100 samples
name = "zero-mean mvnormal"

true_data = Observations.Observation(yt, Γy, name)
```
Currently, the data is retrieved by accessing the stored variables, e.g the fifth data sample is given by `truth_data.samples[5]`, or the covariance matrix by `truth_data.cov`.
