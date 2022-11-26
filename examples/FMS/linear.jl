using LinearAlgebra
using JLD

G = [ 1 2]
u = [ 1; 1]

y = G*u
save("output.jld", "y", y)

