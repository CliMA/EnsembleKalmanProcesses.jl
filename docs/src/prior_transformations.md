# Prior transformation internals

This page isn't necessary for end users of the package: it's intended to 
document TransformedPriors.jl for the benefit of the package maintainers.

## Rationale

We want to make it convenient for users to specify priors on parameters. 
Frequently, a precise distributional form for the prior is unknown (and at any
rate, we should have enough data that the posterior is insensitive to fine detail
in the prior.) Our use case is that the user has an estimate of its expectation
(mean or median) and some notion of spread (standard deviation, inter-quartile range, etc.)

Another necessary consideration is that physical parameter values frequently occupy a restricted domain, while the ensemble Kalman filter and its variants most naturally operate in an unconstrained domain. In addition to specifying the prior distribution, we also need an invertible transformation from the *physical*, constrained coordinates to the unconstrained coordinates in which the algorithm operates.

For simplicity, we assume that the physical parameter domain can be expressed as
a product of one-dimensional intervals, i.e. the domain is contiguous and the
constraints on any parameter are independent of the values of any other parameters.

## Algorithm

### Summary

From the above considerations, the transformation from physical to unconstrained
coordinates needs to deal with as many as five points:

- The lower and upper bounds of the domain,
- The lower and upper bounds of the "spread interval", e.g. the interquartile
  range or ``\mu \pm \sigma``.
- The expectation value, e.g. mean or median.

...

### Three-point transformations

In what follows, we use the following notation:

- ``z`` is the physical, constrained coordinate;
- ``u = \log(w)`` is the unconstrained coordinate.

As is well known, for any two 3-tuples of points on the Riemann sphere, there exists a unique [Möbius transformation](https://en.wikipedia.org/wiki/M%C3%B6bius_transformation) which sends one 3-tuple to the other. We need to send the domain bounds to ``(-\infty, +\infty)``, so the Riemann sphere's one-point
compactification is inappropriate; instead, we find a transformation mapping the
domain to ``[0, \infty)`` and compose that with a ``\log``. In the common case 
where the unconstrained prior is Gaussian, this means we want generalizations of
the [lognormal](https://en.wikipedia.org/wiki/Log-normal_distribution) and "[logit-normal](https://en.wikipedia.org/wiki/Logit-normal_distribution)" distributions.

If we ignore questions of the spread, the map

```math
w(z) = \frac{(z-a)(\mu' - b)}{(z-b)(\mu' - a)}
```

accomplishes the goal of mapping the physical domain ``[a,b]`` to ``u = log(w) = (-\infty, +\infty)`` and ``z = \mu'`` to ``u = 0``, and is well-behaved as ``a \leftarrow -\infty`` or ``b \leftarrow \infty``. 

Consider the distribution

```math
P(u) du = P[\log w(z)] \frac{1}{w(z)} \frac{dw}{dz} dz,
```

in particular the Gaussian case ``u \sim \mathcal{N}(\mu, \sigma)``. It can be verified that ``\mu`` and ``\mu'`` are not algebraically independent: for simplicity, consider the mode ``u_0`` of ``P(u)``. Letting ``\mu'`` be a function of ``\mu``, setting ``\partial u_0 / \partial \mu = 0`` results in a solvable ODE
for ``\mu'(\mu)``. Without loss of generality, we can set ``\mu = 0`` and set 
expectation values through the transformation parameter ``\mu'``.

This argument leaves ``\sigma`` as a free parameter controlling the spread of both the constrained and unconstrained distributions. 

### Five-point transformations

The above transformation is insufficently general: we want to allow the user to specify the spread as an interval. As indicated above, this means we need to generalize the Möbius transformation to one mapping an arbitrary 5-tuple of points to another 5-tuple (in doing so, of course, we destroy the group structure.)

This can be done by generalizing the determinantal definition of the Möbius transformation. Consider the row vector formed from the six monomials

```math
r(z,w) = [1, z, z^2, -w, -w*z, -w*z^2].
```

The transformation is obtained by setting the corresponding polynomial to zero:

```math
p(z,w) = c_1 + c_2 z + c_3 z^2 -c_4 w - c_5 wz - c_6 wz^2 = 0;
```
i.e.

```math
\begin{aligned}
w &= \frac{c_1 + c_2 z + c_3 z^2}{c_4 + c_5 z + c_6 z^2}; \\
z &= - \frac{c_2 - c_5 w \pm \sqrt{(c_2 - c_5 w)^2 - 4(c_1 - c_4 w)(c_3 - c_6 w)}}{2(c_3 + c_6 w)}.
\end{aligned}
```

In the Möbius group, both ``w(z)`` and ``z(w)`` are simple (free of branch cuts); here the best we can do is to make the transformation simple in one direction, which we choose to be ``z(w)``.

We use this form to obtain a map sending source points ``z_1, \ldots, z_5`` to target points ``w_1, \ldots, w_5`` as follows: we define the polymonial as the ``6 \times 6`` determinant

```math
p(z,w) = \det [r(z,w); r(z_1,w_1); r(z_2,w_2); \ldots r(z_5,w_5)].
```

The determinant of a matrix vanishes if any two rows are proportional, which proves ``p(z_i, w_i) = 0`` for ``i = 1, \ldots, 5``. The individual coefficents ``c_1, \ldots, c_6`` can be read off as terms in the expansion of the determinant by minors along the top row.

In fact, we need to nest the expension by minors, because one of the ``w_i`` is always ``\infty`` (the point that the upper bound of the domain interval is mapped to.) We choose this to be ``w_2``, and take the limit ``w_2 \leftarrow \infty`` by expanding each minor by minors again but retaining only the dominant terms (those containing ``w_2``).

The upper or lower bounds of the domain may also be infinite. In principle we could handle this in the same way, by further expanding the determinants, but we opted to simply evaluate the coefficients in these limits and implement the resulting explicit expressions as distinct methods.

### Determining transformation coefficients

It remains to set coefficients (3- or 5- tuples of source and target points) of the transformations above based on user input. Because changing variables in the pdf produces the appropriate Jacobian, the simplest prior specifications to deal with are *quantiles*:

```math
\int^q P(u) du = \int^{u^{-1}(q)} P[\log w(z)] \frac{1}{w(z)} \frac{dw}{dz} dz,
```

so the user can specify a quantile ``q (= \log w_i)`` and its desired position in the constrained distribution ``u^{-1}(q) (= z_i)``, and we can then use the closed-form expressions for the transformations described above to ensure the ``z_i`` is mapped to the corresponding ``w_i``. The "three-point"/Möbius transformation can therefore be used to fix one quantile (in practice, the median) in addition to the domain endpoints, while the "five-point" transform can be used to fix three points (e.g., the median and inter-quartile range.)

Expectation values such as the mean and standard deviation are likely more intuitive to the user; however, no closed-form expression for their values exists, nor is it possible to find a different form for the transformation (of sufficient flexibility) that enables these integrals to be evaluated analytically. In this case, we're forced to evaluate the coefficients numerically: by obtaining expectation values for a given set of coefficients through numerical integration, and then updating coefficients via numerical optimization. This could rapidly become expensive, so we use initial coefficients derived from the closed-form expressions for appropriate quantiles.

...

!!! note "Multi-modality"
    The package only checks that the spread interval is contained within the domain interval. If the requested spread interval is sufficiently wide, the generated transformation will take a unimodal unconstrained distribution to a *bi*-modal constrained distribution. An example is provided by Wikipedia's plot of the
    [logit-normal](https://en.wikipedia.org/wiki/Logit-normal_distribution)" distribution at ``\sigma = 3.16``.

    We regard this as a **feature**, not a bug: because the Kalman algorithm works exclusively in the unconstrained space, performance is unaffected. More complicated transforms could produce cleaner constrained distributions, but no transform can yield something more platykurtic than the uniform distribution on an interval without going multi-modal. In principle the user could be warned when this happens, but the condition for multimodality is a transcendental equation which would be difficult to solve reliably.

