# Now, we define a model which generates a sinusoid given parameters ``\theta``: an
# amplitude and a vertical shift. The model adds a random phase shift upon evaluation.
using Random, Statistics

function sine_random_phase(amplitude, vert_shift; dt = 0.01, rng = Random.GLOBAL_RNG)
    trange = 0:dt:(2 * pi + dt)
    phi = 2 * pi * rand(rng)
    return amplitude * sin.(trange .+ phi) .+ vert_shift
end

# We then define ``G(\theta)``, which returns the observables of the sinusoid
# given a parameter vector. These observables should be defined such that they
# are informative about the parameters we wish to estimate. Here, the two
# observables are the ``y`` range of the curve (which is informative about its
# amplitude), as well as its mean (which is informative about its vertical shift).
function parameter_to_data_map(u; dt = 0.01, rng = Random.GLOBAL_RNG)
    sincurve = sine_random_phase(u["amplitude"], u["vert_shift"], dt = dt, rng = rng)
    return [maximum(sincurve) - minimum(sincurve), mean(sincurve)]
end
